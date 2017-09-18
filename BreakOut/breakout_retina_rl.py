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

import matplotlib.pyplot as plt

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


view_game = True if 1 else False
view_retina = True if 0 else False
record_retina = True if 0 and (not (view_retina or view_game))else False

X_BITS = 8
Y_BITS = 8

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT = 17893
UDP_RETINA = 18881

RET_CFG = {'record': {'voltages': False,
                      'spikes': record_retina,
                      },
           'gabor': False,
           'input_mapping_func': mapf.row_col_to_input,
           'row_bits': X_BITS,
           'lateral_competition': False,
           'direction': False,
           # 'w2s': 20.,
           }
RET_MODE = dvs_modes[MERGED]

delete_prev_run()

# Setup pyNN simulation
sim.setup(timestep=1.0, min_delay=1., max_delay=14.)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp_supervision, 50)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 500)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)

num_rl_neurons = 100
# run_time = 24*60*60*1000.
run_time = 1*60*1000.
tau_plus  = 10.
tau_minus = 12.
tau_c = 1000.
tau_d = 200.
conn_prob = 0.03
w_max = 5.
w_min = 0.


# Create breakout population and activate live output for it
breakout_pop = sim.Population(1, spinn_breakout.Breakout, {}, label="breakout")
if not record_retina and view_game:
    ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT)

retina = Retina(sim, breakout_pop, X_RESOLUTION, Y_RESOLUTION, RET_MODE, cfg=RET_CFG)

left_agent = sim.Population(num_rl_neurons, sim.IF_curr_exp_supervision, {},
                             label='left agent')
right_agent = sim.Population(num_rl_neurons, sim.IF_curr_exp_supervision, {},
                             label='right agent')

direction_left = sim.Population(2, sim.IF_curr_exp, {},
                                label='left direction')
direction_right = sim.Population(2, sim.IF_curr_exp, {},
                                label='right direction')

key_input = sim.Population(3, SpikeInjector, {"port": 12367}, label="key_input")

reward_pop = sim.Population(1, sim.IF_curr_exp, {}, label='reward pop')
punish_pop = sim.Population(1, sim.IF_curr_exp, {}, label='punish pop')

left_noise = sim.Population(num_rl_neurons//2,
                            sim.SpikeSourcePoisson,
                            {'rate': 10.},
                            label='left noise')
right_noise = sim.Population(num_rl_neurons//2,
                            sim.SpikeSourcePoisson,
                            {'rate': 10.},
                            label='right noise')

# Create spike injector to inject keyboard input into simulation

key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])


if view_retina:
    i = 0
    ret_ports_dict = {}
    for ch in retina.pops:
        ret_ports_dict[ch] = {}
        for k in sorted(retina.pops[ch]):

            if 'cs' in k:# and k == 'cs3':

                ex.activate_live_output_for(retina.pops[ch][k]['ganglion'],
                                            host="0.0.0.0", port=UDP_RETINA+i)
                ret_ports_dict[ch][k] = UDP_RETINA + i
                i += 1
    print(ret_ports_dict)

# Connect key spike injector to breakout population
sim.Projection(key_input, breakout_pop, sim.AllToAllConnector(weights=2))

sim.Projection(left_agent, direction_left, sim.AllToAllConnector(weights=0.5),
               label='from left agent to direction left')
sim.Projection(right_agent, direction_right, sim.AllToAllConnector(weights=0.5),
               label='from right agent to direction right')


sim.Projection(direction_left, breakout_pop,
               sim.FromListConnector([(0, 0, 5, 1)]),
               label='from direction left to break_pop')
sim.Projection(direction_right, breakout_pop,
               sim.FromListConnector([(1, 1, 5, 1)]),
               label='from direction right to break_pop')

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

        ret_projs[ch][p] = {}
        ret_projs[ch][p]['left'] = sim.Projection(
                              retina.pops[ch][p]['ganglion'],   left_agent,
                              sim.FixedProbabilityConnector(conn_prob, weights=2.),
                              synapse_dynamics=syn_dyn,
                              label='from retina-{}-{} to left agent'.format(ch, p))
        ret_projs[ch][p]['right'] = sim.Projection(
                                  retina.pops[ch][p]['ganglion'], right_agent,
                                  sim.FixedProbabilityConnector(conn_prob, weights=2.),
                                  synapse_dynamics=syn_dyn,
                                  label='from retina-{}-{} to right agent'.format(ch, p))

sim.Projection(breakout_pop, reward_pop,
               sim.FromListConnector([(0, 0, 5., 1.)]),
               label='from breakout to reward extraction')
sim.Projection(breakout_pop, punish_pop,
               sim.FromListConnector([(1, 0, 5., 1.)]),
               label='from breakout to punishment extraction')

sim.Projection(reward_pop, left_agent,
               sim.AllToAllConnector(weights=0.1),
               target='reward',
               label='from reward extract to left agent')
sim.Projection(reward_pop, right_agent,
               sim.AllToAllConnector(weights=0.1),
               target='reward',
               label='from reward extract to right agent')

sim.Projection(left_noise, left_agent,
               sim.FixedProbabilityConnector(0.1, weights=1.),
               label='from noise left to left agent')

sim.Projection(right_noise, right_agent,
               sim.FixedProbabilityConnector(0.1, weights=1.),
               label='from noise right to right agent')


# Create visualiser
# if not record_retina and view_game:
#     visualiser = Visualiser(
#         UDP_PORT, key_input_connection,
#         x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
#         x_bits=X_BITS, y_bits=Y_BITS)

# if view_retina:
#     print("\n\nView retina :D ----------------------- \n")
#     ret_vis = RetinaVisualiser(ret_ports_dict, retina.shapes)

# Run simulation (non-blocking)
start_time = time.time()
sim.run(run_time)
print("\nTime to execute sim.run() = {:f} minutes".format(
                                     (time.time() - start_time)/60.))

# if view_retina:
#     print("\n\nstarting Retina viewer \n")
#     ret_vis.show()

# Show visualiser (blocking)
# if not record_retina and view_game:
#     visualiser.show()



if record_retina:
    retina_spikes = get_spikes(retina)


    # cam_split_spikes = {}
    # for ch in retina.cam:
    #     cam_split_spikes[ch] = retina.cam[ch].getSpikes(compatible_output=True)

    plot_spikes(retina_spikes)
    dump_compressed(retina_spikes, "game_plus_retina_spikes.pickle")
# End simulation
sim.end()