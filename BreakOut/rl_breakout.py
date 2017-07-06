import spynnaker.pyNN as sim
from breakout_utils import get_punishment_neuron_id, get_reward_neuron_id
from pacman.model.constraints.partitioner_constraints.partitioner_maximum_size_constraint import \
    PartitionerMaximumSizeConstraint
from spynnaker_external_devices_plugin.pyNN.connections. \
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
import spynnaker_external_devices_plugin.pyNN as ex
import spinn_controller
import spinn_breakout
import numpy as np
# from vision.sim_tools.connectors.direction_connectors_rob import    paddle_connection#, direction_connection,subsample_connection,
from vision.sim_tools.connectors.direction_connectors import direction_connection_angle, subsample_connection, \
    paddle_connection
from vision.sim_tools.connectors.mapping_funcs import row_col_to_input_breakout, row_col_to_input_subsamp
import pylab as plt
from vision.spike_tools.vis import plot_output_spikes


def init_pop(label, n_neurons, run_time_ms, machine_timestep_ms):
    print "{} has {} neurons".format(label, n_neurons)
    print "Simulation will run for {}ms at {}ms timesteps".format(
        run_time_ms, machine_timestep_ms)


def send_input(label, sender):
    sender.send_spike(label, 0)

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# Layout of pixels
X_BITS = int(np.ceil(np.log2(X_RESOLUTION)))
Y_BITS = 8  # hard coded to match breakout C implementation

# UDP port to read spikes from
breakout_port = 17893

# Setup pyNN simulation
timestep = 1.
sim.setup(timestep=timestep)

# cell parameters needed for 50Hz rate generator
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 5.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

pyramidal_lif = {'cm': 0.75,
                 'i_offset': 0.0,
                 'tau_m': 20.0,
                 'tau_refrac': 5.0,
                 'tau_syn_E': 5.0,
                 'tau_syn_I': 5.0,
                 'v_reset': -70.0,
                 'v_rest': -65.0,
                 'v_thresh': -50.0
                 }

streamSubset = list()
singleStream = [10]
for i in range(80):
   singleStream.append(i*200 + 10000)
streamSubset.append(singleStream)
singleStream = list()
#for i in range(20):
#   singleStream.append(i*20 + 5000)
streamSubset.append(singleStream)

spikeArray = {'spike_times': streamSubset}
# Instantiate the two executables:
breakout_pop = sim.Population(2, spinn_breakout.Breakout, {}, label="breakout")
#breakout_pop = sim.Population(2, sim.IF_curr_exp, pyramidal_lif, label="breakout")
#controller_pop = sim.Population(2, sim.SpikeSourceArray, spikeArray, {}, label="controller")
controller_pop = sim.Population(3, spinn_controller.RLController, {}, label="controller")

# Connect the executables together (Controller to BreakOut):
sim.Projection(breakout_pop, controller_pop, sim.FromListConnector([(0,0,1,1),(1,1,1,1)]), label='breakOutOut')
sim.Projection(controller_pop, breakout_pop, sim.FromListConnector([(0,0,1,1),(1,1,1,1)]), label='controlOut')

# Create reward/punishment populations (BreakOut to Controller):
reward_pop = sim.Population(1, sim.IF_curr_exp, cell_params_lif, label="reward_pop")
punishment_pop = sim.Population(1, sim.IF_curr_exp, cell_params_lif, label="punishment_pop")
rl_pop = sim.Population(3, sim.IF_curr_exp, cell_params_lif, label="rl_pop")
sim.Projection(controller_pop, rl_pop, sim.FromListConnector([(0,0,2,1),(1,1,2,1), (2,2,2,1)]), label='rl')
sim.Projection(breakout_pop, reward_pop, sim.FromListConnector([(get_reward_neuron_id(), 0, 2, 1)]))
sim.Projection(breakout_pop, punishment_pop, sim.FromListConnector([(get_punishment_neuron_id(), 0, 2, 2)]))
#sim.Projection(reward_pop, controller_pop, sim.FromListConnector([(0, 0, 2, 1)]))
#sim.Projection(punishment_pop, controller_pop, sim.FromListConnector([(0, 1, 2, 1)]))

#rl_pop.record()

#
# Create spike injector to inject keyboard input into simulation
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=breakout_port)
key_input = sim.Population(2, ex.SpikeInjector, {"port": 12367}, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])
key_input_connection.add_start_callback("key_input", send_input)
#

# Create visualiser
visualiser_full = spinn_breakout.Visualiser(
    breakout_port, key_input_connection,
    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
    x_bits=X_BITS, y_bits=Y_BITS)

# Run simulation (non-blocking)
sim.run(None)

# Show visualiser (blocking)
visualiser_full.show()

#spikes = rl_pop.getSpikes()
#print spikes
# End simulation
sim.end()
