import spynnaker.pyNN as sim
from spynnaker_external_devices_plugin.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
import spynnaker_external_devices_plugin.pyNN as ex
import spinn_breakout
import numpy as np
from vision.sim_tools.connectors.direction_connectors import direction_connection, subsample_connection,  paddle_connection
from vision.sim_tools.connectors.mapping_funcs import row_col_to_input_breakout,  row_col_to_input_subsamp
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

#paddle size
paddle_width=8

# Layout of pixels
X_BITS = int(np.ceil(np.log2(X_RESOLUTION)))
Y_BITS =  8#hard coded to match breakout C implementation

subsamp_factor_x=1#8
subsamp_factor_y=127
paddle_row=Y_RESOLUTION-1

X_RES_SUB=X_RESOLUTION/subsamp_factor_x
Y_RES_SUB=Y_RESOLUTION/subsamp_factor_y

subX_BITS= 0#int(np.ceil(np.log2(X_RESOLUTION/subsamp_factor_x)))
subY_BITS=0#int(np.ceil(np.log2(Y_RESOLUTION/subsamp_factor_y)))

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
Delays_medium_vertical=np.round(np.array(delaysList)*pt*subsamp_factor_y*1000)#factoring in subsampling
Delays_medium_horizontal=np.round(np.array(delaysList)*pt*subsamp_factor_x*1000)#factoring in subsampling
Delays_medium_diagonal=np.round(np.array(delaysList)*pt*np.hypot(subsamp_factor_y,subsamp_factor_x)*1000)#factoring in subsampling
Delays_medium_vertical[div-1]=timestep
Delays_medium_horizontal[div-1]=timestep
Delays_medium_diagonal[div-1]=timestep

print "Delays used=",Delays_medium_horizontal

#restrict number of neurons on each core
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

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

#rate generator parameters
rate_weight = 1.5
rate_delay = 16.

# Create breakout population and activate live output for it
breakout_pop = sim.Population(1, spinn_breakout.Breakout, {}, label="breakout")
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=breakout_port)

#Create subsampled pop
breakout_size=  2**(X_BITS+Y_BITS+1)+2
subsamp_size=(X_RES_SUB)*(Y_RES_SUB)
direction_size=subsamp_size#X_RESOLUTION*Y_RESOLUTION
#print "breakout population size: ", breakout_size
#print "subsampled (factor of", subsamp_factor,") population size: ", subsamp_size
subsamp_on_pop=sim.Population(subsamp_size, sim.IF_curr_exp, {}, label="subsample channel on")
subsamp_off_pop=sim.Population(subsamp_size, sim.IF_curr_exp, {}, label="subsample channel off")

paddle_on_pop=sim.Population(X_RES_SUB, sim.IF_curr_exp, {}, label="paddle channel on")
paddle_off_pop=sim.Population(X_RES_SUB, sim.IF_curr_exp, {}, label="paddle channel off")
    
#Create direction detection populations
#n_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="n")
#e_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="e")
#s_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="s")
#w_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="w")
#ne_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="ne")
#nw_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="nw")
#se_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="se")
#sw_pop=sim.Population(subsamp_size,sim.IF_curr_exp,{},  label="sw")

e_on_pop=sim.Population(direction_size,sim.IF_curr_exp,{},  label="e_on")
e_off_pop=sim.Population(direction_size,sim.IF_curr_exp,{},  label="e_off")
w_on_pop=sim.Population(direction_size,sim.IF_curr_exp,{},  label="w_on")
w_off_pop=sim.Population(direction_size,sim.IF_curr_exp,{},  label="w_off")


#activate live outputs
ex.activate_live_output_for(subsamp_on_pop,database_notify_host="localhost",database_notify_port_num=connection_port)
ex.activate_live_output_for(subsamp_off_pop,database_notify_host="localhost",database_notify_port_num=connection_port)
#ex.activate_live_output_for(paddle_on_pop,database_notify_host="localhost",database_notify_port_num=connection_port)
#ex.activate_live_output_for(paddle_off_pop,database_notify_host="localhost",database_notify_port_num=connection_port)

#    
 #Create spike injector to inject keyboard input into simulation
#key_input = sim.Population(2, ex.SpikeInjector, {"port": 12367}, label="key_input")
#key_input_connection = SpynnakerLiveSpikesConnection(receive_labels=None, local_port=19999,send_labels=["key_input"])
#key_input_connection.add_init_callback("key_input", init_pop)
#key_input_connection.add_start_callback("key_input", send_input)

direction_weight=2.7#3.34
paddle_weight=1./paddle_width#direction_weight/paddle_width#0.1

key_input = sim.Population(2, sim.IF_curr_exp, {}, label="key_input") 
key_input_right= sim.Population(1,sim.IF_curr_exp, {}, label="key_input_right")
key_input_left= sim.Population(1,sim.IF_curr_exp, {}, label="key_input_left")

keyright_connections=[(0, 0, 10, 1.)]
keyleft_connections=[(0, 1, 10, 1.)]
sim.Projection(key_input_right, key_input,sim.FromListConnector(keyright_connections))
sim.Projection(key_input_left, key_input,sim.FromListConnector(keyleft_connections))

#make paddle self connections to maintain firing to align with direction population outputs
#sim.Projection(paddle_on_pop, paddle_on_pop,sim.OneToOneConnector(weights=2.9, delays=6.))
#sim.Projection(paddle_on_pop, paddle_on_pop,sim.OneToOneConnector(weights=1, delays=60.))

#create rate generator population
rate_generator = sim.Population(X_RES_SUB, sim.IF_curr_exp, cell_params_lif,
                              label="Rate generation population")
sim.Projection(rate_generator, rate_generator, sim.OneToOneConnector(rate_weight, rate_delay), target='excitatory',
label='rate_pop->rate_pop')

#connect paddle on population to rate_generator
sim.Projection(paddle_on_pop,rate_generator, sim.OneToOneConnector(rate_weight, 1.), target='excitatory')
#connect paddle off population to rate generator
sim.Projection(paddle_off_pop,rate_generator, sim.OneToOneConnector(rate_weight, 1.), target='inhibitory')

#create inline population
inline_east_pop=sim.Population(X_RES_SUB,sim.IF_curr_exp, {}, label="inline east channel on") 
inline_west_pop=sim.Population(X_RES_SUB,sim.IF_curr_exp, {}, label="inline west channel on") 


#connect paddle to paddle population (excitatory)
paddle_inline_east_connections=[]
paddle_inline_west_connections=[]
for i in range(X_RES_SUB):
    for j in range(i+1, X_RES_SUB):
        paddle_inline_west_connections.append((j, i,paddle_weight, 1.))
        paddle_inline_east_connections.append((X_RES_SUB-1-j,X_RES_SUB-1-i, paddle_weight, 1.))
#print 'paddle_east:',  paddle_inline_east_connections
#print 'paddle_west:',  paddle_inline_west_connections

#sim.Projection(paddle_on_pop,inline_east_pop,sim.FromListConnector(paddle_inline_east_connections))
#sim.Projection(paddle_on_pop,inline_west_pop,sim.FromListConnector(paddle_inline_west_connections))
sim.Projection(rate_generator,inline_east_pop,sim.FromListConnector(paddle_inline_east_connections))
sim.Projection(rate_generator,inline_west_pop,sim.FromListConnector(paddle_inline_west_connections))

#connect east to inline population (excitatory)
sim.Projection(e_on_pop,inline_east_pop, sim.OneToOneConnector(weights=direction_weight))
sim.Projection(w_on_pop,inline_west_pop, sim.OneToOneConnector(weights=direction_weight))

#connect inline to key input right population 
sim.Projection(inline_east_pop, key_input_right, sim.AllToAllConnector(weights=10.))
sim.Projection(inline_west_pop, key_input_left, sim.AllToAllConnector(weights=10.))

#connect east to key_input population (excitatory)
#sim.Projection(e_on_pop, key_input_right, sim.AllToAllConnector(weights=test_weight))
#sim.Projection(w_on_pop, key_input_left, sim.AllToAllConnector(weights=test_weight))

#if east pop is firing then fire key_input to the right
#when paddle and east pop are inline stop firing (inhibit key_input to the right) right
#excitatory connections between east pop and key_input right 
#inhibitory connections between key input 

#declare projection weights 
#breakout--> subsample
subweight=5#2.7
#subsample-->direction connections -- needs to take into account div
direction_weight=2.6
print "Weights used=",direction_weight
         
#generate connection lists
#[Connections_se_on, Connections_se_off]=direction_connection("south east", X_RESOLUTION, Y_RESOLUTION, div, Delays_medium, weight, row_col_to_input_breakout)
#[Connections_subsamp_on,Connections_subsamp_off]=subsample_connection(X_RESOLUTION, Y_RESOLUTION, subsamp_factor_x,subsamp_factor_y, subweight,row_col_to_input_subsamp)
[Connections_e_on, Connections_e_off]=direction_connection("east", X_RES_SUB,Y_RES_SUB,\
                                                                                                        div, Delays_medium_horizontal, direction_weight, row_col_to_input_subsamp)
                                                                                                        
[Connections_w_on, Connections_w_off]=direction_connection("west", X_RES_SUB,Y_RES_SUB,\
                                                                                div, Delays_medium_horizontal, direction_weight, row_col_to_input_subsamp)
                                                                                

[Connections_subsamp_on,Connections_subsamp_off]=subsample_connection(X_RESOLUTION, Y_RESOLUTION-1, subsamp_factor_x,subsamp_factor_y, \
                                                                                                                                subweight,row_col_to_input_breakout)
                                                                                                                                
#subweight=5#10
[Connections_paddle_on,Connections_paddle_off]=paddle_connection(X_RESOLUTION,paddle_row, subsamp_factor_x, subweight, row_col_to_input_breakout)


#Connections_n=direction_connection("north", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
#Connections_e=direction_connection("east", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
#Connections_s=direction_connection("south", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
#Connections_w=direction_connection("west", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
#Connections_ne=direction_connection("north east", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
#Connections_nw=direction_connection("north west", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
#Connections_se=direction_connection("south east", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
#Connections_sw=direction_connection("south west", X_RESOLUTION/subsamp_factor, Y_RESOLUTION/subsamp_factor, div, Delays_medium, weight)
      
# Connect key spike injector to breakout population
sim.Projection(key_input, breakout_pop, sim.OneToOneConnector(weights=5))

#Connect breakout population to direction populations
#projectionSe_on=sim.Projection(breakout_pop,se_on_pop,sim.FromListConnector(Connections_se_on))
#projectionSe_off=sim.Projection(breakout_pop,se_off_pop,sim.FromListConnector(Connections_se_off))
projectionSub_on=sim.Projection(breakout_pop,subsamp_on_pop,sim.FromListConnector(Connections_subsamp_on))
projectionSub_off=sim.Projection(breakout_pop,subsamp_off_pop,sim.FromListConnector(Connections_subsamp_off))

projectionPaddle_on=sim.Projection(breakout_pop,paddle_on_pop,sim.FromListConnector(Connections_paddle_on))
projectionPaddle_off=sim.Projection(breakout_pop,paddle_off_pop,sim.FromListConnector(Connections_paddle_off))

#Connect direction populations to subsampled populations
#projectionSub_on=sim.Projection(se_on_pop,subsamp_on_pop,sim.FromListConnector(Connections_subsamp_on))
#projectionSub_off=sim.Projection(se_off_pop,subsamp_off_pop,sim.FromListConnector(Connections_subsamp_off))

projectionE_on=sim.Projection(subsamp_on_pop,e_on_pop,sim.FromListConnector(Connections_e_on))
projectionE_off=sim.Projection(subsamp_off_pop,e_off_pop,sim.FromListConnector(Connections_e_off))
projectionW_on=sim.Projection(subsamp_on_pop,w_on_pop,sim.FromListConnector(Connections_w_on))
projectionW_off=sim.Projection(subsamp_off_pop,w_off_pop,sim.FromListConnector(Connections_w_off))

#Connect subsample population to direction populations
#projectionN=sim.Projection(subsamp_pop,n_pop,sim.FromListConnector(Connections_n))
#projectionE=sim.Projection(subsamp_pop,e_pop,sim.FromListConnector(Connections_e))
#projectionS=sim.Projection(subsamp_pop,s_pop,sim.FromListConnector(Connections_s))
#projectionW=sim.Projection(subsamp_pop,w_pop,sim.FromListConnector(Connections_w))
#projectionNe=sim.Projection(subsamp_pop,ne_pop,sim.FromListConnector(Connections_ne))
#projectionNw=sim.Projection(subsamp_pop,nw_pop,sim.FromListConnector(Connections_nw))
#projectionSe=sim.Projection(subsamp_pop,se_pop,sim.FromListConnector(Connections_se))
#projectionSw=sim.Projection(subsamp_pop,sw_pop,sim.FromListConnector(Connections_sw))

#Create Live spikes Connections
live_output_connection=SpynnakerLiveSpikesConnection(
   # receive_labels=["subsample channel on","subsample channel off", "ne","nw","se", "sw","n", "e", "s", "w"], local_port=connection_port, send_labels=None)
   receive_labels=["subsample channel on","subsample channel off", "paddle channel on","paddle channel off", "e_on","e_off"], local_port=connection_port, send_labels=None)

 #Create visualisers
#visualiser_full = spinn_breakout.Visualiser(
#    breakout_port, key_input_connection,
#    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
#    x_bits=X_BITS, y_bits=Y_BITS)
#
visualiser_full = spinn_breakout.Visualiser(
    breakout_port, None,
    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
    x_bits=X_BITS, y_bits=Y_BITS)
#
#visualiser_sub = spinn_breakout.Visualiser_subsamp(
#    None, live_output_connection, "subsample channel on","subsample channel off", 
#    x_res=X_RES_SUB, y_res=Y_RES_SUB,
#    x_bits=subX_BITS, y_bits=subY_BITS)
#    
#visualiser_paddle = spinn_breakout.Visualiser_subsamp(
#    key_input_connection, live_output_connection, "paddle channel on","paddle channel off", 
#    x_res=X_RES_SUB, y_res=1,
#    x_bits=subX_BITS, y_bits=subY_BITS)
   
#subsamp_on_pop.record('spikes')
#e_on_pop.record('spikes')
#w_on_pop.record('spikes')
#inline_east_pop.record('spikes')
#inline_west_pop.record('spikes')
#key_input.record('spikes')
#paddle_on_pop.record('spikes')
#rate_generator.record('spikes')

# Run simulation (non-blocking)
sim.run(None)
#sim.run(simulationTime)
#subsamp_spks=subsamp_on_pop.getSpikes()
#e_spks=e_on_pop.getSpikes()
#inline_east_spikes=inline_east_pop.getSpikes()
#w_spks=w_on_pop.getSpikes()
#inline_west_spikes=inline_west_pop.getSpikes()
#key_spikes=key_input.getSpikes()
#paddle_spikes=paddle_on_pop.getSpikes()
#rate_spikes=rate_generator.getSpikes()

# Show visualiser (blocking)

visualiser_full.show()
#visualiser_sub.show()
#visualiser_paddle.show()
#visualiser_e.show()
#visualiser_se.show()

# End simulation
sim.end()

plt.figure()               
plot_output_spikes(subsamp_spks,plotter=plt)
plt.xlim([0,simulationTime])
plt.figure()       
plot_output_spikes(e_spks,plotter=plt)
plt.xlim([0,simulationTime])
plt.figure()       
plot_output_spikes(w_spks,plotter=plt)
plt.xlim([0,simulationTime])
plt.figure()       
plot_output_spikes(paddle_spikes,plotter=plt)
plt.xlim([0,simulationTime])
plt.figure()       
plot_output_spikes(rate_spikes,plotter=plt)
plt.xlim([0,simulationTime])
plt.figure()       
plot_output_spikes(inline_east_spikes,plotter=plt)
plt.xlim([0,simulationTime])
plt.figure()       
plot_output_spikes(inline_west_spikes,plotter=plt)
plt.xlim([0,simulationTime])
plt.figure()       
plot_output_spikes(key_spikes,plotter=plt)
plt.xlim([0,simulationTime])
plt.show()
