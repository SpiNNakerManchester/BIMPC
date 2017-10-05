import spynnaker7.pyNN as sim
from spynnaker7.pyNN.external_devices import SpynnakerLiveSpikesConnection

from spynnaker7.pyNN.external_devices import SpynnakerExternalDevicePluginManager as ex

import time
import spinn_breakout

# Layout of pixels
from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
from spinn_breakout.visualiser.visualiser import Visualiser
from vision.retina import Retina, dvs_modes, MERGED
from vision.sim_tools import dump_compressed
from vision.sim_tools.connectors import mapping_funcs as mapf
from vision.sim_tools.vis import RetinaVisualiser
from vision.spike_tools.vis.vis_tools import plot_in_out_spikes
import numpy as np
import matplotlib.pyplot as plt
from random_balanced_network import RandomBalancedNetwork
from control_output import ControlOutput

def get_weights(projs, loop, weights, w_init):
    output_fname = "weights_loop_{:06d}.txt".format(loop)
    outf = open(output_fname, 'w')

    for ch in projs:
        # if ch not in weights:
        #     weights[ch] = {}

        for p in projs[ch]:
            # if p not in weights[ch]:
            #     weights[ch][p] = {}

            for a in projs[ch][p]:
                # if a not in weights[ch][p]:
                #     weights[ch][p][a] = []

                ws = projs[ch][p][a].getWeights(format='array')
                outf.write("--- Weights from {:s} - {:s} to {:} ---\n\n".format(ch, p, a))
                for r in range(ws.shape[0]):
                    for c in range(ws.shape[1]):
                        if not np.isnan(ws[r, c]):
                            outf.write("{:05d}, {:05d}, {:010.6f}\n".
                                       format(r, c, ws[r, c]))

                outf.write(
                    "\n--- ----------------------------------------------- ---\n\n")
                # mws = np.memmap("weights_{}_{}_{}_{}_mmap.npy".format(ch, p, a, loop),
                #               shape=ws[np.where(~np.isnan(ws))].shape, dtype='float32',
                #               mode='w+')
                # mws[:] = ws[np.where(~np.isnan(ws))]
                # weights[ch][p][a].append(mws)

    outf.close()

    return weights


def plot_weights(weights):
    for ch in weights:
        for p in weights[ch]:
            for a in weights[ch][p]:
                fig = plt.figure()
                ax = plt.subplot(1,1,1)
                loops = len(weights[ch][p][a])
                plt.plot(weights[ch][p][a])
                plt.margins(0.1, 0.1)
                plt.draw()
                plt.savefig("weights_loop_{}_channel_{}_filter_{}_agent_{}.pdf".
                            format(loops, ch, p, a))
                plt.close(fig)


def get_spikes(retina):
    spikes = {}
    for ch in retina.conns.keys():
        spikes[ch] = {}
        spikes[ch]['cam'] = retina.cam[ch].getSpikes(compatible_output=True)

        for pop in retina.conns[ch].keys():
            bipolar = retina.pops[ch][pop]['bipolar']. \
                getSpikes(compatible_output=True)
            ganglion = retina.pops[ch][pop]['ganglion']. \
                getSpikes(compatible_output=True)
            inter = retina.pops[ch][pop]['inter']. \
                getSpikes(compatible_output=True)

            spikes[ch][pop] = {'bipolar': bipolar, 'inter': inter,
                               'ganglion': ganglion}

    return spikes


def delete_prev_run():
    import os
    files = os.listdir(os.getcwd())
    for file in files:
        if file.endswith(".png") or file.endswith(".m4v") or \
           file.endswith(".bz2") or file.endswith(".pdf"):

            os.remove(os.path.join(os.getcwd(), file))


def plot_spikes(spikes_dict, cam_split_spikes=None):
    if cam_split_spikes is not None:
        for ch in cam_split_spikes:
            print("plotting cam %s"%ch)
            in_color = 'magenta' if ch == 'off' else 'cyan'
            plot_in_out_spikes([], cam_split_spikes[ch],
                               'game_retina_cam_%s.pdf'%ch,
                               in_color, None, None)
    for ch in spikes_dict:
        in_color = 'magenta' if ch == 'off' else 'cyan'

        for pop in spikes_dict[ch]:
            if pop == 'cam':
                print("plotting cam %s (%d)"
                      % (ch, len(spikes_dict[ch][pop])))
                in_color = 'magenta' if ch == 'off' else 'cyan'
                plot_in_out_spikes([], spikes_dict[ch][pop],
                                   'game_retina_cam_%s.pdf' % ch,
                                   in_color, None, None)

                continue

            if 'bipolar' in spikes_dict[ch][pop]:
                in_spk = spikes_dict[ch][pop]['bipolar']
            else:
                in_spk = []

            if 'ganglion' in spikes_dict[ch][pop]:
                out_spk = spikes_dict[ch][pop]['ganglion']
            else:
                out_spk = []

            print("\nlen %s-%s in %d, out %d"%(ch, pop, len(in_spk), len(out_spk)))
            plot_in_out_spikes(in_spk, out_spk,
                               'game_retina_%s_%s.pdf'%(ch, pop),
                               in_color, pop, ch)


def plot_ctrl_spikes(dir_left, dir_right, agent_left, agent_right, ms=4, nctrl=7):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 2, 1)
    al = np.array(agent_left)
    plt.plot(al[:, 1], al[:, 0], 'rx', label='Agent L', markersize=ms)
    ar = np.array(agent_right)
    plt.plot(ar[:, 1], ar[:, 0], 'g+', label='Agent R', markersize=ms)
    plt.margins(0.1, 0.1)
    plt.grid()
    lgd0 = plt.legend(bbox_to_anchor=(1., 1.15), loc='upper right',
                     ncol=2)
    plt.draw()

    ax = plt.subplot(1, 2, 2)
    dl = np.array(dir_left)
    rows = np.where(dl[:,0] == 0)[0]
    plt.plot(dl[rows, 1], dl[rows, 0], 'rx', label='Control L (%u)'%len(rows),
             markersize=ms)
    dr = np.array(dir_right)
    rows = np.where(dr[:, 0] == 1)[0]
    plt.plot(dr[rows, 1], 1 + dr[rows, 0], 'g+', label='Control R (%u)'%len(rows),
             markersize=ms)
    plt.margins(0.1, 0.1)
    plt.grid()
    lgd1 = plt.legend(bbox_to_anchor=(1., 1.15), loc='upper right',
                     ncol=2)
    plt.draw()
    plt.savefig("agents_and_control_spikes.pdf",
                bbox_extra_artists=(lgd0,lgd1,), bbox_inches='tight')
    # plt.show()

def switch(v):
    return (True if v else False)

def all_to_one_conn(num_pre, post_id, w, d):
    return [(pre, post_id, w, d) for pre in range(num_pre)]

########################################################################
########################################################################
########################################################################

X_BITS = 8
Y_BITS = 8

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT = 17893
UDP_RETINA = 18881

RAND_BAL_CFG = {'input_noise_rate': 5.,
       'exc_noise_rate': 250., 'inh_noise_rate': 250.,'w_noise': 0.32,
       'exc_noise_start': 1., 'inh_noise_start': 0.,
       'w_exc': {'prob': 0.1, 'mu': 0.4, 'sigma': 0.1, 'low': 0, 'high': 20.},
       'd_exc': {'prob': 0.1, 'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
       'w_inh': {'prob': 0.1, 'mu': 2.0, 'sigma': 0.1, 'low': 0, 'high': 20.},
       'd_inh': {'prob': 0.1, 'mu': 0.75, 'sigma': 0.375, 'low': 1., 'high': 14.4},
       'w_in': {'mu': 0.001, 'sigma': 0.01, 'low': 0, 'high': 20.},
       'd_in': {'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
      }

RET_CFG = {'record': {'voltages': False,
                      'spikes': False,
                      },
           'gabor': False,
           'input_mapping_func': mapf.row_col_to_input,
           'row_bits': X_BITS,
           'lateral_competition': False,
           'direction': False,
           # 'w2s': 20.,
           }

RET_MODE = dvs_modes[MERGED]

LEFT_CTRL_CFG = dict(w2s=5., w_keep_alive = 5., w_control_inh = 5.*0.25,
                     w_inh_feedback = 5.*2., d_keep_alive = 2,
                     d_inh_feedback = 20, label='LEFT control')

RIGHT_CTRL_CFG = dict(w2s=5., w_keep_alive = 5., w_control_inh = 5.*0.25,
                      w_inh_feedback = 5.*2., d_keep_alive = 2,
                      d_inh_feedback = 20, label='RIGHT control')

########################################################################
########################################################################
########################################################################

do_full_run = switch(1)

view_game = switch(1)
view_retina = switch(0)
record_retina = switch( 0 and (not (view_retina or view_game)) )
# record_control = switch(1 and not view_game)
record_control = switch(1 and (not do_full_run))

use_punishment = switch(0)

use_wta = switch(1)
retrieve_weights = switch(0 or do_full_run)
rand_w = switch(0)
# controller_keep_alive = switch(1)
controller_inh = switch(1)
opposing_inh = switch(0)

use_rbn = switch(0)
use_retina = switch(1)

########################################################################
########################################################################
########################################################################



delete_prev_run()

# Setup pyNN simulation
sim.setup(timestep=1.0, min_delay=1., max_delay=144.)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp_supervision, 50)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 60)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 500)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)

if do_full_run:
    hours = 2
    n_loops = 3 * hours
    run_time = hours * 60 * 60 * 1000.
else:
    n_loops = 1
    if record_control:
        run_time = n_loops * 1 * 60 *1000.
    else:
        run_time = n_loops * 20 * 60 * 1000.

num_rl_neurons = 100
num_noise = num_rl_neurons//2
num_rbn = 300
num_left_ctrl = 2
num_right_ctrl = 2
tau_plus  = 10.
tau_minus = 12.
tau_c = 1000.
tau_d = 200.
noise_rate = 10
# conn_prob = 0.03
conn_prob = 0.3
noise_conn_prob = 0.1
inh_conn_prob = 0.1
w2s = 4.8
agent_w2s = 6.98125

if not use_rbn:
    w_noise = agent_w2s*(1./13.)
else:
    w_noise = agent_w2s*(1./20.)
w_rbn = agent_w2s*(1./20.)
w_max = agent_w2s*1.5
w_min = 0.
w_reward = 0.001
w_punish = 0.001
w_init = agent_w2s*(0.25)
w_inh  = w2s*(0.25)
w_opp_inh  = w2s*(0.25)
w_ctrl = w2s*(1./1.0)
w_keep_alive = w2s*(0.9)
w_wta = agent_w2s*(0.80)
# ######################################################################
#  P  O  P  U  L  A  T  I  O  N  S
# ######################################################################

# Create breakout population and activate live output for it
breakout_pop = sim.Population(1, spinn_breakout.Breakout, {}, label="breakout")
if view_game:
    ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT)

if use_retina:
    retina = Retina(sim, breakout_pop, X_RESOLUTION, Y_RESOLUTION, RET_MODE, cfg=RET_CFG)

left_agent = sim.Population(num_rl_neurons, sim.IF_curr_exp_supervision,
                            {'tau_syn_E': 5.0, 'tau_syn_I': 3.0,},
                            label='left agent')


right_agent = sim.Population(num_rl_neurons, sim.IF_curr_exp_supervision,
                             {'tau_syn_E': 5.0, 'tau_syn_I': 3.0,},
                             label='right agent')


left_control = ControlOutput(sim, num_left_ctrl, LEFT_CTRL_CFG)
right_control = ControlOutput(sim, num_right_ctrl, RIGHT_CTRL_CFG)

# left_control = sim.Population(1, sim.SpikeSourceArray,
#                               {'spike_times': [[0, 1, 2, 40]]},
#                               label='testing left control')

if record_control:
    left_agent.record()
    right_agent.record()
    left_control.output.record()
    right_control.output.record()

reward_pop = sim.Population(1, sim.IF_curr_exp, {}, label='reward pop')
punish_pop = sim.Population(1, sim.IF_curr_exp, {}, label='punish pop')

left_noise = sim.Population(num_noise, sim.SpikeSourcePoisson,
                            {'rate': noise_rate}, label='left noise')
right_noise = sim.Population(num_noise, sim.SpikeSourcePoisson,
                            {'rate': noise_rate}, label='right noise')
if use_rbn:
    left_rbn = RandomBalancedNetwork(sim, num_rbn, cfg=RAND_BAL_CFG)
    right_rbn = RandomBalancedNetwork(sim, num_rbn, cfg=RAND_BAL_CFG)

if use_wta:
    left_wta_inh = sim.Population(1, sim.IF_curr_exp, {},
                                  label='left wta inh neuron')
    right_wta_inh = sim.Population(1, sim.IF_curr_exp, {},
                                   label='right wta inh neuron')

# Create spike injector to inject keyboard input into simulation
# key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])


# ######################################################################
#  P  R  O  J  E  C  T  I  O  N  S
# ######################################################################

sim.Projection(left_agent, left_control.input,
               sim.FromListConnector(
                        all_to_one_conn(num_rl_neurons, 0, w_ctrl, 1)),
               label='from left agent to direction left')

sim.Projection(right_agent, right_control.input,
               sim.FromListConnector(
                        all_to_one_conn(num_rl_neurons, 1, w_ctrl, 1)),
               label='from right agent to direction right')

##################
##################

sim.Projection(left_control.output, breakout_pop,
               sim.FromListConnector([(0, 0, 1, 1.)]),
               label='from direction left to break_pop')

sim.Projection(right_control.output, breakout_pop,
               sim.FromListConnector([(1, 0, 1, 1.)]),
               label='from direction right to break_pop')

##################
##################
if use_retina:
    ret_projs = {}
    for ch in retina.pops:
        ret_projs[ch] = {}
        for p in retina.pops[ch]:
            if 'cam' in p:
                continue

            time_dep   = sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus,
                                           tau_c=tau_c, tau_d=tau_d)
            weight_dep = sim.AdditiveWeightDependence(w_min=w_min, w_max=w_max,)
            stdp = sim.STDPMechanism(time_dep, weight_dep,
                                     neuromodulation=True)
            syn_dyn = sim.SynapseDynamics(slow=stdp)

            if rand_w:
                exc_weights = sim.RandomDistribution('uniform',
                                                     [0.5*w_init, w_init])
            else:
                exc_weights = w_init

            ret_projs[ch][p] = {}
            ret_projs[ch][p]['left'] = sim.Projection(
                                  retina.pops[ch][p]['ganglion'],   left_agent,
                                  sim.FixedProbabilityConnector(conn_prob,
                                      weights=exc_weights, generate_on_machine=True),
                                  synapse_dynamics=syn_dyn,
                                  target='excitatory',
                                  label='from retina-{}-{} to left agent'.format(ch, p))

            ret_projs[ch][p]['right'] = sim.Projection(
                                      retina.pops[ch][p]['ganglion'], right_agent,
                                      sim.FixedProbabilityConnector(conn_prob,
                                          weights=exc_weights, generate_on_machine=True),
                                      synapse_dynamics=syn_dyn,
                                      target='excitatory',
                                      label='from retina-{}-{} to right agent'.format(ch, p))

sim.Projection(breakout_pop, reward_pop,
               sim.FromListConnector([(0, 0, w2s, 2.)]),
               label='from breakout to reward extraction')
sim.Projection(breakout_pop, punish_pop,
               sim.FromListConnector([(1, 0, w2s, 2.)]),
               label='from breakout to punishment extraction')

# reward connections

sim.Projection(reward_pop, left_agent,
               sim.AllToAllConnector(weights=w_reward,
                                     generate_on_machine=True),
               target='reward',
               label='from reward extract to left agent')
sim.Projection(reward_pop, right_agent,
               sim.AllToAllConnector(weights=w_reward,
                                     generate_on_machine=True),
               target='reward',
               label='from reward extract to right agent')

# punishment connections
if use_punishment:
    sim.Projection(punish_pop, left_agent,
                   sim.AllToAllConnector(weights=w_punish,
                                         generate_on_machine=True),
                   target='punishment',
                   label='from punishment extract to left agent')
    sim.Projection(punish_pop, right_agent,
                   sim.AllToAllConnector(weights=w_punish,
                                         generate_on_machine=True),
                   target='punishment',
                   label='from punishment extract to right agent')


# noise connections

sim.Projection(left_noise, left_agent,
               sim.FixedProbabilityConnector(noise_conn_prob,
                     weights=w_noise, generate_on_machine=True),
               label='from noise left to left agent')

sim.Projection(right_noise, right_agent,
               sim.FixedProbabilityConnector(noise_conn_prob,
                     weights=w_noise, generate_on_machine=True),
               label='from noise right to right agent')

if use_rbn:
    sim.Projection(left_rbn.output, left_agent,
                   sim.FixedProbabilityConnector(noise_conn_prob,
                         weights=w_rbn, generate_on_machine=True),
                   label='from left RBN to left agent'
                  )

    sim.Projection(right_rbn.output, right_agent,
                   sim.FixedProbabilityConnector(noise_conn_prob,
                         weights=w_rbn, generate_on_machine=True),
                   label='from left RBN to left agent'
                  )


# inhibitory connections between left & right agents/directions
if opposing_inh:
    sim.Projection(left_agent, right_agent,
                   # sim.FixedProbabilityConnector(inh_conn_prob,
                   #       weights=w_inh, generate_on_machine=True),
                   sim.AllToAllConnector(
                            weights=w_opp_inh, generate_on_machine=True),
                   label='left agent inhibits right agent',
                   target='inhibitory')

    sim.Projection(right_agent, left_agent,
                   # sim.FixedProbabilityConnector(inh_conn_prob,
                   #       weights=w_inh, generate_on_machine=True),
                   sim.AllToAllConnector(
                           weights=w_opp_inh, generate_on_machine=True),
                   label='right agent inhibits left agent',
                   target='inhibitory')

if use_wta:
    sim.Projection(left_agent, left_wta_inh,
                   sim.AllToAllConnector(w2s,
                         allow_self_connections=True, generate_on_machine=True),
                   target='excitatory',
                   label='left agent to wta inh neuron')
    sim.Projection(left_wta_inh, left_agent,
                   sim.AllToAllConnector(w_wta,
                         allow_self_connections=True, generate_on_machine=True),
                   target='inhibitory',
                   label='wta inh neuron left agent')

    sim.Projection(right_agent, right_wta_inh,
                   sim.AllToAllConnector(w2s,
                         allow_self_connections=True, generate_on_machine=True),
                   target='excitatory',
                   label='right agent to wta inh neuron')
    sim.Projection(right_wta_inh, right_agent,
                   sim.AllToAllConnector(agent_w2s,
                         allow_self_connections=True, generate_on_machine=True),
                   target='inhibitory',
                   label='wta inh neuron left agent')

if controller_inh:
    sim.Projection(left_control.output, right_control.output,
                   sim.OneToOneConnector(weights=w2s,
                                         generate_on_machine=False),
                   label='left dir inhibits right dir',
                   target='inhibitory')

    sim.Projection(right_control.output, left_control.output,
                   sim.OneToOneConnector(weights=w2s,
                                         generate_on_machine=False),
                   label='right dir inhibits left dir',
                   target='inhibitory')


start_time = 0.
recorded_weights = []
weights = {}

for loop in range(n_loops):
    print("\n\nRUNNING Loop %d/%d (%d ms)\n"%(loop+1, n_loops, run_time//n_loops))
    start_time = time.time()
    sim.run(run_time//n_loops)
    print("\nTime to execute sim.run() = {:f} minutes".format(
                                         (time.time() - start_time)/60.))

    if retrieve_weights:
        weights = get_weights(ret_projs, loop, weights, w_init)

    # plot_weights(weights)
    if record_control:
        ctrl_left = left_control.output.getSpikes(compatible_output=True)
        ctrl_right = right_control.output.getSpikes(compatible_output=True)
        agent_left = left_agent.getSpikes(compatible_output=True)
        agent_right = right_agent.getSpikes(compatible_output=True)

    # if record_control:
        plot_ctrl_spikes(ctrl_left, ctrl_right, agent_left, agent_right,
                         ms=8, nctrl=num_left_ctrl)
        plt.show()
# End simulation
sim.end()

