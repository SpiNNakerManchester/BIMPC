import spynnaker.pyNN as sim
from spynnaker_external_devices_plugin.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
import spynnaker_external_devices_plugin.pyNN as ex
import spinn_breakout
import numpy as np
from vision.sim_tools.connectors.direction_connectors import direction_connection, subsample_connection
from vision.sim_tools.connectors.mapping_funcs import row_col_to_input_breakout
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
Y_BITS =  8#hard coded to match breakout C implementation

subsamp_factor=8

subX_BITS= int(np.ceil(np.log2(X_RESOLUTION/subsamp_factor)))
subY_BITS=int(np.ceil(np.log2(Y_RESOLUTION/subsamp_factor)))

#pixel speed 
pps=70#
#movement timestep
pt=1./pps 
#number of pixels to calculate speed across
div=2

# UDP ports to read spikes from
breakout_port=17893
connection_port=19993

# Setup pyNN simulation
timestep=1.
sim.setup(timestep=timestep)
simulationTime=3000

#setup speed delays
delaysList=range(div-1,-1,-1)
Delays_medium=np.round(np.array(delaysList)*pt*subsamp_factor*1000)#factoring in subsampling
Delays_medium[div-1]=timestep
print "Delays used=",Delays_medium

#restrict number of neurons on each core
#sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)

# Create breakout population and activate live output for it
breakout_pop = sim.Population(1, spinn_breakout.Breakout, {}, label="breakout")
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=breakout_port)

#Create subsampled pop
breakout_size=  2**(X_BITS+Y_BITS+1)+2
subsamp_size=(X_RESOLUTION/subsamp_factor)*(Y_RESOLUTION/subsamp_factor)
print "breakout population size: ", breakout_size
print "subsampled (factor of", subsamp_factor,") population size: ", subsamp_size
subsamp_on_pop=sim.Population(subsamp_size, sim.IF_curr_exp, {}, label="subsample channel on")
subsamp_off_pop=sim.Population(subsamp_size, sim.IF_curr_exp, {}, label="subsample channel off")

#activate live outputs
ex.activate_live_output_for(subsamp_on_pop,database_notify_host="localhost",database_notify_port_num=connection_port)
ex.activate_live_output_for(subsamp_off_pop,database_notify_host="localhost",database_notify_port_num=connection_port)
    
 #Create spike injector to inject keyboard input into simulation
key_input = sim.Population(2, ex.SpikeInjector, {"port": 12367}, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(receive_labels=None, local_port=19999,send_labels=["key_input"])
key_input_connection.add_init_callback("key_input", init_pop)
key_input_connection.add_start_callback("key_input", send_input)

#declare projection weights 
#breakout--> subsample
subweight=2.7
#subsample-->direction connections -- needs to take into account div
weight=2.5
print "Weights used=",weight
         
#generate connection lists
[Connections_subsamp_on,Connections_subsamp_off]=subsample_connection(X_RESOLUTION, Y_RESOLUTION, subsamp_factor, subweight,row_col_to_input_breakout)
#      
# Connect key spike injector to breakout population
sim.Projection(key_input, breakout_pop, sim.OneToOneConnector(weights=5))

#Connect breakout population to subsample population
projectionSub=sim.Projection(breakout_pop,subsamp_on_pop,sim.FromListConnector(Connections_subsamp_on))
projectionSub=sim.Projection(breakout_pop,subsamp_off_pop,sim.FromListConnector(Connections_subsamp_off))

#Create Live spikes Connections
sub_output_connection=SpynnakerLiveSpikesConnection(
    receive_labels=["subsample channel on","subsample channel off"], local_port=connection_port, send_labels=None)

 #Create visualisers
#visualiser_full = spinn_breakout.Visualiser(
#    breakout_port, key_input_connection,
#    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
#    x_bits=X_BITS, y_bits=Y_BITS)
#
visualiser_sub = spinn_breakout.Visualiser_subsamp(
    key_input_connection, sub_output_connection, "subsample channel on","subsample channel off", 
    x_res=X_RESOLUTION/subsamp_factor, y_res=Y_RESOLUTION/subsamp_factor,
    x_bits=subX_BITS, y_bits=subY_BITS)


#subsamp_pop.record('spikes')

# Run simulation (non-blocking)
sim.run(None)
#sim.run(simulationTime)
#subsamp_spks=subsamp_pop.getSpikes()


# Show visualiser (blocking)

#visualiser_full.show()
visualiser_sub.show()

# End simulation
sim.end()

#plt.figure()               
#plot_output_spikes(subsamp_spks,plotter=plt)
#plt.xlim([0,simulationTime])
#plt.figure()       
#plot_output_spikes(se_spks,plotter=plt)
#plt.xlim([0,simulationTime])
#plot_output_spikes(right_spks,plotter=plt)
#plt.xlim([0,simulationTime])
#plt.show()
