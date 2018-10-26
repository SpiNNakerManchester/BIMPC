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
from vision.spike_tools.vis.vis_tools import plot_in_out_spikes, video_from_spike_array, \
                        imgs_in_T_from_spike_array, images_to_video

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

def plot_spikes(spikes_dict, cam_split_spikes=None,
                produce_video=False, shapes_dict=None):
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

    if produce_video:
        for pop in spikes_dict['on']:
            # off_spikes = spikes_dict['off'][pop]['ganglion']
            on_spikes = spikes_dict['on'][pop]['ganglion']

            video_from_spike_array(on_spikes, shapes_dict[pop]['width'],
                                   shapes_dict[pop]['height'],
                                   0, 100000000, 20,
                                   out_array=True, fps=50, up_down=1,
                                   title='test_game_with_retina_%s'%pop)

view_game = True if 0 else False
view_retina = True if 0 else False
record_retina = True if 1 and (not (view_retina or view_game))else False

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
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 500)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 1000)

# Create breakout population and activate live output for it
breakout_pop = sim.Population(1, spinn_breakout.Breakout, {}, label="breakout")
if not record_retina and view_game:
    ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT)

retina = Retina(sim, breakout_pop, X_RESOLUTION, Y_RESOLUTION, RET_MODE, cfg=RET_CFG)



run_time = 10000. if record_retina else None


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

# Create spike injector to inject keyboard input into simulation
# key_input = sim.Population(3, SpikeInjector, {"port": 12367}, label="key_input")
# if not record_retina:
#     key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])
#
# # Connect key spike injector to breakout population
# sim.Projection(key_input, breakout_pop, sim.AllToAllConnector(weights=2))

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

    plot_spikes(retina_spikes, produce_video=False, shapes_dict=retina.shapes)
    dump_compressed(retina_spikes, "game_plus_retina_spikes.pickle")
# End simulation
sim.end()