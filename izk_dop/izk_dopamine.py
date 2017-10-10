import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from pyNN.random import NumpyRNG, RandomDistribution
import matplotlib.pyplot as plt
import time
import sys
import pickle
import copy
import bz2
import glob
import os

import spynnaker7.pyNN as sim
# import spynnaker_extra_pynn_models as q
# sim = None

#from https://github.com/NEvision/NE15/blob/master/poisson/poisson_tools.py


def poisson_generator(rate, t_start, t_stop):
    '''Poisson train generator
       :param rate: The rate at which a neuron will fire (Hz)
       :param t_start: When should the neuron start to fire (milliseconds)
       :param t_stop: When should the neuron stop firing (milliseconds)
       
       :returns: Poisson train firing at rate, from t_start to t_stop (milliseconds)
    '''
    np.random.seed()
    isi = np.random.poisson(rate, (t_stop - t_start)//rate)
    ts = np.clip(t_start + np.cumsum(isi), t_start, t_stop)

    return np.sort(ts)


def new_poisson_times(rate, t_start, t_stop, num_neurons):
    return [ poisson_generator(rate, t_start, t_stop) \
                               for _ in range(num_neurons) ]


def dump_compressed(data, name):
    with bz2.BZ2File('%s.bz2'%name, 'w') as f:
        pickle.dump(data, f)

def load_compressed(name):
    with bz2.BZ2File('%s.bz2'%name, 'w') as f:
        obj = pickle.load(f)
        return obj

def trunc_weights(w):
    fix_point_one = float((1 << 16))
    tw = w * fix_point_one
    iw = np.uint32(tw)
    
    nw = np.float32(iw >> 16)
    nw += np.float32( (iw & 0xFFFF)/fix_point_one )
    return nw

def nml(v):
    mx = np.max(v)
    if mx == 0:
        return np.zeros_like(v)
    else:
        return (v.copy())/mx

def remove_ticks(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def seed_rand(seed=None):
    # if seed is None:
    #     time.sleep(0.001)
    #     time.sleep(np.random.random()*0.1)
    #     seed = np.uint32(time.time()*(10*10))

    np.random.seed(seed)

def izk_connectivity(num_exc, num_inh, conn_prob=0.10, exc_delay=5.,
                     exc_weight=4., inh_weight=-4., e2e_w_scale=0.2,
                     sd=0.001, num_syn_sd=1.):
    e2e_ws = e2e_w_scale
    n_neurons = num_exc + num_inh
    num_synapses = int(np.round(n_neurons*conn_prob))
    nids = np.arange(n_neurons)
    e2e = []; e2i = []; i2e = []; i2i = []
    e_delay = exc_delay
    i_delay = 1

    for src in range(num_exc):
        seed_rand()
        num_post = int(np.round( np.random.normal(loc=num_synapses,
                                                  scale=num_syn_sd) ))
        seed_rand()
        post = np.random.choice(nids, size=num_post, replace=False)

        exc = post[np.where( np.logical_and(post <  num_exc, \
                                            post != src) )] #no self connections
        seed_rand()
        ew = np.random.uniform(size=exc.size)*exc_weight*e2e_ws
        ed = np.random.randint(1, exc_delay+1, size=exc.size)
        # ew = np.random.normal(exc_weight*e2e_ws, sd, size=exc.size)

        inh = post[np.where(post >= num_exc)] - num_exc
        seed_rand()
        iw = np.random.uniform(size=inh.size)*exc_weight
        # iw = np.random.normal(exc_weight, sd, size=inh.size)

        e2e += [(src, dst, ew[i], ed[i])
                                       for i, dst in enumerate(exc)]
        e2i += [(src, dst, iw[i], i_delay)
                                       for i, dst in enumerate(inh)]

    for src in range(num_inh):
        seed_rand()
        num_post = int(np.round( np.random.normal(loc=num_synapses,
                                                  scale=num_syn_sd) ))
        seed_rand()
        post = np.random.choice(nids, size=num_post, replace=False)
        exc  = post[np.where(post < num_exc)]
        iw   = np.random.normal(inh_weight, sd, size=exc.size)
        i2e += [(src, dst, iw[i], i_delay)
                                        for i, dst in enumerate(exc)]
        inh  = post[np.where(post >= num_exc)] - num_exc
        iw   = np.random.normal(inh_weight, sd, size=inh.size)
        i2i += [(src, dst, iw[i], i_delay)
                                        for i, dst in enumerate(inh)]


    return {'exc_to_exc': e2e, 'exc_to_inh': e2i, 
            'inh_to_exc': i2e, 'inh_to_inh': i2i}


def groups_connectivity(num_exc, num_inh, group_ids, weight=4.):
    g2e = []; g2i = []
    delay = 1
    for src in group_ids:
        exc = group_ids[src][np.where(group_ids[src] <  num_exc)]
        inh = group_ids[src][np.where(group_ids[src] >= num_exc)] - num_exc

        g2e += [(src, dst, weight, delay) for dst in exc]
        g2i += [(src, dst, weight, delay) for dst in inh]

    return {'group_to_exc': g2e, 'group_to_inh': g2i}


neurons_per_core = 10
sources_per_core = 50
neurons_mult = 10.

ms = 1000.
time_step = 1
tstep_ms = int(ms*time_step)

debug         = True if 0 else False
poisson_stim  = True if 1 else False
stim_inh      = True if 1 else False
groups_stim   = True if 1 else False
dopamine_stim = True if 1 else False
izk_neurons   = True if 0 else False
do_stdp       = True if 1 else False
e2e_loop      = True if 1 else False
short_distal  = True if 0 else False
threshold_weight_change = True if 1 else False

#from polychronization paper
tau_plus  = 10.
tau_minus = 12.
a_plus  = 0. # NOT TAKEN INTO ACCOUNT
a_minus = 0. # NOT TAKEN INTO ACCOUNT
# a_plus = 1.
# a_minus = 1.
# tau_plus  = 20.
# tau_minus = 20.
# a_plus = 0.1
# a_minus = 0.12

#from distal reward paper

if short_distal:
    tau_c = 50.
    tau_d = 10.
else:
    tau_c = 1000.
    tau_d = 200.

#from Mantas' script
# tau_plus = 2.
# tau_minus = 1.
# tau_c = 20.0
# tau_d = 5.0

conn_prob = 0.10
e_delay = 2.
# num_neurons = 50
# num_neurons = 200
# num_neurons = 500
# neurons_mult = 20.
num_neurons = 1000*neurons_mult
num_exc = int(num_neurons*0.8)
num_inh = int(num_neurons - num_exc)
num_groups = int(num_neurons*0.1)
group_conn_prob = 0.05
neurons_per_group = int(num_neurons*group_conn_prob)
groups_spike_per_sec = 5

min_dopamine_delay = e_delay + 3
max_dopamine_delay = 1000#tau_c//2 #ms

poisson_rate = int( min(1., 10./neurons_mult) ) #Hz

max_w_scale = 1.01
max_w_scale = (1./1.5)
max_w_scale = (1./8.2625)
max_w_scale = (1./8.)
max_w_scale = (1./(6.*neurons_mult))
if izk_neurons:
    w2s   = 24.5
    w_max = w2s*max_w_scale
else:
    w2s   = 3.725
    w_max = w2s*max_w_scale

w_min  = 0.
# psn_w  =  w2s*1.0
# psn_w  =  w2s*(1./1.8)
psn_w  =  w2s*(1./2.)
w_init =  w2s*(1./(20.*neurons_mult)) if 1 else 0. #~ X pre for post to spike
w_inh  = -w2s*(1./(10.*neurons_mult)) if 1 else 0. #~ X pre for post to spike

#reduce e2e weights to avoid crazy feedback loops
#in function this works as w_init*e2e_ws
e2e_w_scale = 0.5#0.2125
e2e_w_scale = 1.#0.2125
dw_thresh = 0.0005

# increase w2s to make the groups spike consistently
grp_w2s = w2s*1.01
grp_w2s = w2s*1.0
grp_w2s = w2s*0.99

dopamine_weight = 0.0001
dopamine_weight = 0.0005
dopamine_weight = 0.00075
group_spike_dt = 110
NO_CONNECTION = 0.
gen_markersize = 1.
loc_markersize = 1.
n_loops = 1
hours_to_sim = 0.5
# hours_to_sim = 10./60.
hours   = 0.
minutes = 0.
seconds = 60.
real_time = int( np.ceil( tstep_ms*(hours*60*60 + minutes*60 + seconds) ) )
sim_time = int( real_time + 1000 )

plot_dt = 1000




def get_avg_weight(weights):
    
    n_weights = np.sum(~ np.isnan(weights) )
    
    return np.sum(weights[~ np.isnan(weights)])/n_weights


def get_not_group_weights(all_group_ids, group_id, e2e_w, e2i_w, num_exc, num_inh):
    ids = all_group_ids[group_id]
    # print("get_group_weights: ")
    # print(ids)
    exc_group_ids = ids[np.where(ids <  num_exc)]
    inh_group_ids = ids[np.where(ids >= num_exc)] - num_exc
    exc_ids = np.where( np.arange(num_exc) != exc_group_ids )[0]
    inh_ids = np.where( np.arange(num_inh) != inh_group_ids )[0]

    e2e_g0 = e2e_w[exc_ids, exc_ids].copy()
    e2i_g0 = e2i_w[exc_ids, inh_ids].copy()
    # e2e_g1 = e2e_w[exc_ids, :].copy()
    # e2e_g_w = np.ones_like(e2e_w)*np.nan
    # e2e_g_w[:, exc_ids] = e2e_g0
    # e2e_g_w[exc_ids, :] = e2e_g1
    # print(e2e_g_w)
    # print(e2i_g_w)
    return e2e_g0, e2i_g0


def get_group_weights(all_group_ids, group_id, e2e_w, e2i_w, num_exc, num_inh):
    ids = all_group_ids[group_id]
    # print("get_group_weights: ")
    # print(ids)
    exc_ids = np.int32(ids[np.where(ids <  num_exc)])
    inh_ids = np.int32(ids[np.where(ids >= num_exc)] - num_exc)
    not_exc_ids = np.where( np.arange(num_exc) != exc_ids )[0]
    not_inh_ids = np.where( np.arange(num_inh) != inh_ids )[0]

    e2e_g0 = e2e_w[exc_ids, :].copy()
    e2e_g1 = e2e_w[not_exc_ids, exc_ids].copy()
    
    e2i_g0 = e2i_w[exc_ids, :].copy()
    e2i_g1 = e2i_w[not_exc_ids, inh_ids].copy()
    
    fname = "weights_%03d_exc_to_exc_g_w.npy"%(np.random.randint(0, 999))
    e2e_g_w = np.memmap(os.path.join(os.getcwd(), fname), 
                           dtype='float16', mode='w+',
                           shape=e2e_w.shape)
    e2e_g_w.fill(np.nan)
    # e2e_g_w = np.ones_like(e2e_w)*np.nan
    e2e_g_w[exc_ids, :] = e2e_g0
    e2e_g_w[not_exc_ids, exc_ids] = e2e_g1

    fname = "weights_%03d_exc_to_exc_g_w.npy"%(np.random.randint(0, 999))
    e2i_g_w = np.memmap(os.path.join(os.getcwd(), fname), 
                           dtype='float16', mode='w+',
                           shape=e2i_w.shape)
    e2i_g_w.fill(np.nan)
    # e2i_g_w = np.ones_like(e2i_w)*np.nan
    e2i_g_w[exc_ids, :] = e2i_g0
    e2i_g_w[not_exc_ids, inh_ids] = e2i_g1
    
    # print(e2e_g_w)
    # print(e2i_g_w)
    return e2e_g_w, e2i_g_w


def new_group_and_dopamine_times(num_groups, sim_time, time_step, groups_spike_per_sec,
                                 group_spike_dt, min_dop_delay=3, max_dop_delay=100):
    
    dopamine_times = [[]]
    group_times = [[] for i in range(num_groups)]
    
    time_for_groups = groups_spike_per_sec*group_spike_dt
    time_remaining = time_step - time_for_groups
    allowed_rand_per_group = int(np.ceil(float(time_remaining)/groups_spike_per_sec))
    

    for tm in range(0, sim_time, time_step): #go through sim time in 1 second batches
        # print("From %d to %d"%(tm, tm+time_step))
        seed_rand()
        gids = np.random.choice(range(0, num_groups),
                                size=groups_spike_per_sec, replace=False)





        #groups spike at least dt ms appart
        seed_rand()
        times = tm + np.random.choice(range(0, int(time_for_groups + time_remaining*0.25)),
                                      size=len(gids), replace=False)
        times[:] = np.sort(times)
        # print("TIMES")
        # print(times)

        for i in range(len(times) - 1):
            dt = times[i+1] - times[i]
            if dt < group_spike_dt:
                remaining_dt = group_spike_dt - dt
                seed_rand()
                times[i+1] += np.round( np.random.uniform(remaining_dt, 
                                                remaining_dt + allowed_rand_per_group) )
                
                if times[i+1] > tm + time_step:
                    times[i+1] = tm + time_step - 1
        # print("After adjustment")
        # print(times)

        for i in range(len(gids)):
            if gids[i] == 0:
                seed_rand()
                dop_time = times[i] + np.round( np.random.uniform(min_dop_delay, 
                                                                  max_dop_delay) )
                if dop_time > tm + time_step:
                    dop_time = tm + time_step - 1

                print("dopamine times", dop_time)
                dopamine_times[0].append(dop_time)

            group_times[gids[i]].append(times[i])
    
    # sys.exit()

    return group_times, dopamine_times


def update_main_conns(prev_e2e, prev_e2i, prev_i2e, prev_i2i):
    
    e2e = []
    for r in range(len(prev_e2e)):
        for c in range(len(prev_e2e[0])):
            if np.isnan(prev_e2e[r, c]):
                continue
            e2e.append((r, c, prev_e2e[r, c], 1))

    e2i = []
    for r in range(len(prev_e2i)):
        for c in range(len(prev_e2i[0])):
            if np.isnan(prev_e2i[r, c]):
                continue
            e2i.append((r, c, prev_e2i[r, c], 1))

    return {'exc_to_exc': e2e, 'exc_to_inh': e2i, 
            'inh_to_exc': prev_i2e, 'inh_to_inh': prev_i2i}




# e2e_w0 = trunc_weights(e2e_w0)
# e2i_w0 = trunc_weights(e2i_w0)

# print("-----------------------------------------------------")
# print(e2e_w0)
# print("-----------------------------------------------------")
# print(e2i_w0)
# print("-----------------------------------------------------")

#########################################################################
# S I M U L A T O R    S E T U P
#########################################################################

if izk_neurons:
    if do_stdp:
        cell = sim.IZK_curr_exp_supervision
    else:
        cell = sim.IZK_curr_exp

    exc_params = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8,
                  'v_init': -65, 'u_init': -65*0.2,
                  'tau_syn_E': 1.0, 'tau_syn_I': 2.0}
    inh_params = {'a': 0.1,  'b': 0.2, 'c': -65, 'd': 2,
                  'v_init': -65, 'u_init': -65*0.2,
                  'tau_syn_E': 1.0, 'tau_syn_I': 1.0}
else:
    if do_stdp:
        cell = sim.IF_curr_exp_supervision
    else:
        cell = sim.IF_curr_exp

    exc_params = { 'cm': 0.3,         'i_offset': 0.005,  'tau_m': 10.0,
                   'tau_refrac': 4.0, 'tau_syn_E': 1.,  'tau_syn_I': 1.,
                   'v_reset': -70.0,  'v_rest': -65.0,  'v_thresh': -55.4
                  }
    inh_params = {'cm': 0.3,          'i_offset': 0.0,  'tau_m': 10.0,
                  'tau_refrac': 2.0,  'tau_syn_E': 1.,  'tau_syn_I': 1.,
                  'v_reset': -70.0,   'v_rest': -65.0,  'v_thresh': -56.4
                 }
inh_pois = []
exc_pois = []
avg_weights = []
s0_weights = []
not_s0_weights = []

def get_avg_of_arrays(arr1, arr2):
    return ( (arr1.sum() + arr2.sum())/(arr1.size + arr2.size) )



python_rng = NumpyRNG()

num_loops = n_loops if minutes == 0 else int( np.round( (hours_to_sim*60.)/minutes ) ) 
# num_loops = 1
spikes = {}
for loop in range(num_loops):
    print("\n")
    print( "\tLOOP %d / %d (%d neurons @ %d neurons per core) -----------------"%
           (loop+1, num_loops, num_neurons, neurons_per_core) )
    print("\n")
    
    ########################################################################
    # S P I K E    T I M E S    G E N E R A T I O N
    ########################################################################
    #check random generators so it mostly does not repeat neurons in diff groups
    #dopamine should spike whenever group 0 spikes

    group_times, dopamine_times = new_group_and_dopamine_times(num_groups, real_time,
                                      tstep_ms, groups_spike_per_sec, group_spike_dt,
                                      min_dop_delay=min_dopamine_delay,
                                      max_dop_delay=max_dopamine_delay)

    if debug:
        for g in range(num_groups):
            group_times[g][:] = sorted(group_times[g])
            print("Group %d spikes at times"%g)
            print(group_times[g])

    sim.setup(timestep=time_step)

    sim.set_number_of_neurons_per_core(cell, neurons_per_core)
    sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, sources_per_core)
    sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, sources_per_core)

    ########################################################################
    # P O P U L A T I O N S
    ########################################################################

    if poisson_stim:
        poisson_pops = {} # split to use 1-to-1 connector with main pops
        poisson_pops['exc'] = sim.Population(num_exc, 
                                             sim.SpikeSourcePoisson,
                                             {'rate': poisson_rate,
                                              'start': 0, 'duration': real_time},
                                              label='Poisson (EXC)')
        # exc_pois[:] = new_poisson_times(poisson_rate, 0, sim_time, num_exc)
        # poisson_pops['exc'] = sim.Population(num_exc, 
                                             # sim.SpikeSourceArray,
                                             # {'spike_times': exc_pois,},
                                              # label='Poisson (EXC)')

        if stim_inh:
            # inh_pois[:] = new_poisson_times(poisson_rate, 0, sim_time, num_inh)
            # poisson_pops['inh'] = sim.Population(num_inh, sim.SpikeSourceArray,
                                                 # {'spike_times': inh_pois},
                                                 # label='Poisson (INH)')
            poisson_pops['inh'] = sim.Population(num_inh, 
                                                 sim.SpikeSourcePoisson,
                                                 {'rate': poisson_rate,
                                                  'start': 0, 'duration': real_time},
                                                  label='Poisson (INH)')
    main_pop = {}
    main_pop['exc'] = sim.Population(num_exc, cell, exc_params, label='Main (EXC)')
    main_pop['exc'].record()
    main_pop['inh'] = sim.Population(num_inh, cell, inh_params, label='Main (INH)')
    main_pop['inh'].record()

    if dopamine_stim and do_stdp:
        dopamine_pop = sim.Population(1, sim.SpikeSourceArray,
                                      {'spike_times': dopamine_times},
                                      label='Dopamine')

    if groups_stim:
        groups_pop = sim.Population(num_groups, sim.SpikeSourceArray,
                                    {'spike_times': group_times},
                                    label='Group stimulations',)



    ########################################################################
    # P R O J E C T I O N S
    ########################################################################

    # if loop > 0:
    #     main_conns.clear()
    #     main_conns = update_main_conns(prev_e2e, prev_e2i, prev_i2e, prev_i2i)


    if poisson_stim:
        print("Generating Poisson process spikes")
        psn_to_exc = sim.Projection(poisson_pops['exc'], main_pop['exc'],
                                    sim.OneToOneConnector(weights=psn_w,
                                        generate_on_machine=True),
                                    target='excitatory',
                                    )
        if stim_inh:
            psn_to_inh = sim.Projection(poisson_pops['inh'], main_pop['inh'],
                                        sim.OneToOneConnector(weights=psn_w,
                                            generate_on_machine=True),
                                        target='excitatory')

    #add stdp to this
    if do_stdp:
        # modulation = True if dopamine_stim else False

        time_dep   = sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus,
                                       tau_c=tau_c, tau_d=tau_d)
        weight_dep = sim.AdditiveWeightDependence(w_min=w_min, w_max=w_max,
                                                  A_plus=a_plus, A_minus=a_minus)
        stdp = sim.STDPMechanism(time_dep, weight_dep, 
                                 neuromodulation=True)
        syn_dyn = sim.SynapseDynamics(slow=stdp)
        print("************ ------------ using stdp ------------ ***********")
        print("tau_plus = %f\ttau_minus = %f\ttau_c = %f\ttau_d = %f"%
              (tau_plus, tau_minus, tau_c, tau_d))
        print("w_min = %f\tw_max = %f\ta_plus = %f\ta_minus = %f"%
              (w_min, w_max, a_plus, a_minus))
    else:
        syn_dyn = None


    if e2e_loop:
        e2e_rnd_w = RandomDistribution('uniform', (0., w_init), rng=python_rng)
        exc_to_exc = sim.Projection(main_pop['exc'], main_pop['exc'],
                                    sim.FixedProbabilityConnector(conn_prob,
                                        weights=e2e_rnd_w,
                                        generate_on_machine=True),
                                    target='excitatory',
                                    synapse_dynamics=syn_dyn,
                                    )

    exc_to_inh = sim.Projection(main_pop['exc'], main_pop['inh'],
                                sim.FixedProbabilityConnector(conn_prob,
                                        weights=e2e_rnd_w,
                                        generate_on_machine=True),
                                target='excitatory',
                                synapse_dynamics=syn_dyn,
                                )

    #no stdp on inh to X

    inh_to_exc = sim.Projection(main_pop['inh'], main_pop['exc'],
                                sim.FixedProbabilityConnector(conn_prob,
                                        weights=w_inh,
                                        generate_on_machine=True),
                                target='inhibitory')
    inh_to_inh = sim.Projection(main_pop['inh'], main_pop['inh'],
                                sim.FixedProbabilityConnector(conn_prob,
                                        weights=w2s,
                                        generate_on_machine=True),
                                target='inhibitory')

    #add the dopamine target
    if dopamine_stim and do_stdp:
        dope_to_exc = sim.Projection(dopamine_pop, main_pop['exc'],
                                     sim.AllToAllConnector(weights=dopamine_weight,
                                        generate_on_machine=True),
                                     target='reward')
        dope_to_inh = sim.Projection(dopamine_pop, main_pop['inh'],
                                     sim.AllToAllConnector(weights=dopamine_weight,
                                        generate_on_machine=True),
                                     target='reward')

    #stimulate groups
    if groups_stim:
        groups_to_exc = sim.Projection(groups_pop, main_pop['exc'],
                                       sim.FixedProbabilityConnector(
                                        group_conn_prob*0.8,
                                        weights=grp_w2s,
                                        generate_on_machine=True),
                                       target='excitatory')
        groups_to_inh = sim.Projection(groups_pop, main_pop['inh'],
                                       sim.FixedProbabilityConnector(
                                        group_conn_prob*0.2,
                                        weights=grp_w2s,
                                        generate_on_machine=True),
                                       target='excitatory')


    ########################################################################
    # R U N   S I M U L A T I O N
    ########################################################################

    th = sim_time//(1000*60*60)
    tm = (sim_time - th*1000*60*60)//(1000*60)
    ts = (sim_time - th*1000*60*60 - tm*1000*60)/1000.
    print("\n\n ----------------------------------------------------------")
    print("\n Run time = %d h: %d m ; %f s "%(th, tm, ts))
    print("\n ----------------------------------------------------------\n\n")
           
    sim.run(sim_time)

    #--- read outputs
    
    spikes.clear()
    spikes['exc'] = main_pop['exc'].getSpikes(compatible_output=True)
    spikes['inh'] = main_pop['inh'].getSpikes(compatible_output=True)

    sim.end()


    ########################################################################
    # P L O T    R E S U L T S
    ########################################################################

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    # Dopamine times 
    for t in dopamine_times[0]:
        t += ((loop + 1)*sim_time)//1000
        plt.plot([t, t], [0, num_neurons], 'c--', lw=1)

    # Spikes
    try:
        nids  = [nid for (nid, spkt) in spikes['exc']]
        spkts = [spkt for (nid, spkt) in spikes['exc']]
        plt.plot(spkts, nids, '.b', markersize=gen_markersize)
    except:
        print("unable to plot exc spikes - global")
        pass
    
    try:
        nids  = [nid + num_exc for (nid, spkt) in spikes['inh']]
        spkts = [spkt for (nid, spkt) in spikes['inh']]
        plt.plot(spkts, nids, '.r', markersize=gen_markersize)
    except:
        print("unable to plot inh spikes - global")
        pass


    plt.margins(0.1, 0.1)
    plt.draw()
    plt.savefig("spike_activity_loop_%03d_of_%03d.png"%(loop, num_loops), dpi=300)
    plt.close(fig)





    try:
        # spikes['exc'].sort(key=lambda tup: tup[1])
        spikes['exc'][:] = spikes['exc'][spikes['exc'][:, 1].argsort()]
    except:
        print("exc sorted from spikes error")
        pass
    
    try:
        spikes['inh'][:] = spikes['inh'][spikes['inh'][:, 1].argsort()]
    except:
        print("inh sorted from spikes error")
        pass

    ms_to_s = 1./1000.
    start_dop_idx = 0
    start_exc_idx = 0
    start_inh_idx = 0
    nids  = []
    spkts = []
    for start_t in range(0, sim_time, plot_dt):
        end_t = start_t + plot_dt
        
        spike_type = 'regular'
        # Dopamine times 
        for t in dopamine_times[0]:
            if start_t <= t and t < end_t:
                spike_type = 'DOPAMINE'
                break

        if spike_type == 'regular':
            np.random.seed()
            coin_flip = np.random.uniform(0., 1.)
            if coin_flip < 0.5:
                continue
        
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        # Dopamine times 
        for t in dopamine_times[0][start_dop_idx:]:
            if start_t <= t and t < end_t:
                t = (loop*sim_time + t)*ms_to_s #convert to seconds
                plt.plot([t, t], [0, num_neurons], 'c--', lw=0.5)
                start_dop_idx += 1
                plt.text(t-0.01, -30., 'Dopamine', fontsize=8)

        # Spikes
        nids[:]  = []
        spkts[:] = []
        try:
            for (nid, spkt) in spikes['exc'][start_exc_idx:]:
                if start_t <= spkt  and spkt < end_t:
                    nids.append(nid)
                    spkts.append(spkt)
                    start_exc_idx += 1

            plt.plot((loop*sim_time + np.array(spkts))*ms_to_s, nids, 
                     '.b', markersize=loc_markersize)
        except:
            print("per second exc spike plot error")
            pass


        nids[:]  = []
        spkts[:] = []
        try:
            for (nid, spkt) in spikes['inh'][start_inh_idx:]:
                if start_t <= spkt  and spkt < end_t:
                    nids.append(nid + num_exc)
                    spkts.append(spkt)
                    start_inh_idx += 1

            plt.plot((loop*sim_time + np.array(spkts))*ms_to_s, nids, 
                      '.r', markersize=loc_markersize)
        except:
            print("per second inh spike plot error")
            pass


        plt.ylabel('Neuron Id')
        plt.xlabel('Time (s)')

        plt.margins(0.1, 0.1)
        plt.draw()

        if not os.path.isdir(os.path.join(os.getcwd(), 'spike_activity_images')):
            os.makedirs(os.path.join(os.getcwd(), 'spike_activity_images'))

        plt.savefig("spike_activity_images/spike_activity_%s_from_sec_%06d_-_to_sec_%06d.png"%
                (spike_type, (loop*sim_time + start_t)//1000, (loop*sim_time + end_t)//1000),
                    dpi=300)

        plt.close(fig)
