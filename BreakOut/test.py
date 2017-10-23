import spynnaker7.pyNN as sim


from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
import spinn_breakout

# Layout of pixels
from spynnaker.pyNN.models.utility_models.spike_injector import SpikeInjector
from spinn_breakout.visualiser.visualiser import Visualiser

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

# Connect key spike injector to breakout population
sim.Projection(key_input, breakout_pop, sim.AllToAllConnector(weights=2))

# Create visualiser
visualiser = Visualiser(
    UDP_PORT, key_input_connection,
    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
    x_bits=X_BITS, y_bits=Y_BITS)

# Run simulation (non-blocking)
sim.run(None)

# Show visualiser (blocking)
visualiser.show()


# End simulation
sim.end()
