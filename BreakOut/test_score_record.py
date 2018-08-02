import spynnaker7.pyNN as sim
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
import spinn_breakout
from spinn_front_end_common.utilities.globals_variables import get_simulator

# Layout of pixels
from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
from spinn_breakout.visualiser.visualiser import Visualiser
import numpy as np

def get_scores(breakout_pop,simulator):
    b_vertex = breakout_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

    return scores.tolist()

X_BITS = 8
Y_BITS = 8


# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT = 17893

# Setup pyNN simulation
sim.setup(timestep=1.0)

# Create breakout population and activate live output for it
breakout_pop = sim.Population(1, spinn_breakout.Breakout, {}, label="breakout")
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT)

# Create spike injector to inject keyboard input into simulation
key_input = sim.Population(2, SpikeInjector, {"port": 12367}, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])
#
# # Connect key spike injector to breakout population
sim.Projection(key_input, breakout_pop, sim.AllToAllConnector(weights=2))

# Create visualiser
# visualiser = Visualiser(
#     UDP_PORT, key_input_connection,
#     x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
#     x_bits=X_BITS, y_bits=Y_BITS)

# Run simulation (non-blocking)
simulator = get_simulator()

sim_duration = 10*60*1000.

sim.run(sim_duration)
scores=get_scores(breakout_pop=breakout_pop, simulator=simulator)

# Show visualiser (blocking)
#visualiser.show()

print scores
# End simulation
sim.end()
