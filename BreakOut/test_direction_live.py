import spynnaker.pyNN as sim
from spynnaker_external_devices_plugin.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
import spynnaker_external_devices_plugin.pyNN as ex
import spinn_breakout
import numpy as np
#from vision.sim_tools.connectors.direction_connectors_rob import    paddle_connection#, direction_connection,subsample_connection,
from vision.sim_tools.connectors.direction_connectors import direction_connection_angle, subsample_connection, paddle_connection
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

# Layout of pixels
X_BITS = int(np.ceil(np.log2(X_RESOLUTION)))
Y_BITS =  8#hard coded to match breakout C implementation

paddle_row=Y_RESOLUTION-1

#paddle size
paddle_width=8
#pixel speed 
pps=50.#speed for 20ms frame rate in bkout.c
#movement timestep
pt=1./pps 
#number of pixels to calculate speed across
div=2

# UDP port to read spikes from
breakout_port=17893

# Setup pyNN simulation
timestep=1.
sim.setup(timestep=timestep)

#restrict number of neurons on each core
#sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

#cell parameters needed for 50Hz rate generator
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
#print "breakout population size: ", breakout_size
#print "subsampled (factor of", subsamp_factor,") population size: ", subsamp_size
horz_subsamp_on_pop=sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="subsample channel on")
horz_subsamp_off_pop=sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="subsample channel off")
vert_subsamp_on_pop=sim.Population(Y_RESOLUTION-1, sim.IF_curr_exp, {}, label="vertical subsample channel on")
vert_subsamp_off_pop=sim.Population(Y_RESOLUTION-1, sim.IF_curr_exp, {}, label="vertical subsample channel off")

#Create paddle detection populations
paddle_on_pop=sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="paddle channel on")
paddle_off_pop=sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="paddle channel off")

#create rate generator population (for constant paddle output)
rate_generator = sim.Population(X_RESOLUTION, sim.IF_curr_exp, cell_params_lif,
                              label="Rate generation population")
sim.Projection(rate_generator, rate_generator, sim.OneToOneConnector(rate_weight, rate_delay), target='excitatory',
label='rate_pop->rate_pop')

#create east and west direction populations   
e_on_pop=sim.Population(X_RESOLUTION,sim.IF_curr_exp,{},  label="e_on")
e_off_pop=sim.Population(X_RESOLUTION,sim.IF_curr_exp,{},  label="e_off")
w_on_pop=sim.Population(X_RESOLUTION,sim.IF_curr_exp,{},  label="w_on")
w_off_pop=sim.Population(X_RESOLUTION,sim.IF_curr_exp,{},  label="w_off")

n_on_pop=sim.Population(Y_RESOLUTION-1,sim.IF_curr_exp,{},  label="n_on")
n_off_pop=sim.Population(Y_RESOLUTION-1,sim.IF_curr_exp,{},  label="n_off")
s_on_pop=sim.Population(Y_RESOLUTION-1,sim.IF_curr_exp,{},  label="s_on")
s_off_pop=sim.Population(Y_RESOLUTION-1,sim.IF_curr_exp,{},  label="s_off")

#create inline population
inline_east_pop=sim.Population(X_RESOLUTION,sim.IF_curr_exp, {}, label="inline east channel") 
inline_west_pop=sim.Population(X_RESOLUTION,sim.IF_curr_exp, {}, label="inline west channel") 

#Create key input population
key_input = sim.Population(2, sim.IF_curr_exp, {}, label="key_input") 
#Create key input right and left populations
key_input_right= sim.Population(1,sim.IF_curr_exp, {}, label="key_input_right")
key_input_left= sim.Population(1,sim.IF_curr_exp, {}, label="key_input_left")

#declare projection weights 
#breakout--> subsample
subweight=5#2.7
#subsample-->direction connections 
sub_direction_weight=2.5
inline_direction_weight=2.7#3.34
inline_paddle_weight=1./paddle_width#inline_direction_weight/paddle_width

#generate breakout to subsample connection lists (note Y-RESOLUTION-1 to ignore paddle row)
[Connections_horz_subsamp_on,Connections_horz_subsamp_off]=subsample_connection(X_RESOLUTION, Y_RESOLUTION-1, 1,Y_RESOLUTION-1, \
                                                                                                                                subweight,row_col_to_input_breakout)
                                                                                                                                
[Connections_vert_subsamp_on,Connections_vert_subsamp_off]=subsample_connection(X_RESOLUTION, Y_RESOLUTION-1, X_RESOLUTION,1, \
                                                                                                                                subweight,row_col_to_input_breakout)
                                                                                                                                
#generate breakout to paddle connection lists
[Connections_paddle_on,Connections_paddle_off]=paddle_connection(X_RESOLUTION,paddle_row, 1, subweight, row_col_to_input_breakout)

#generate subsample to direction connection lists
dist=1    
angle=8
dir_delay=int(pt*1000)

#TODO: make these non-uniform distributions, larger weights in the centre of image?
[Connections_e_on,_],[Connections_e_off,_]=direction_connection_angle("E", \
                                                angle,
                                                dist, 
                                                X_RESOLUTION, 1, 
                                                row_col_to_input_subsamp,                       
                                                0,1,                     
                                                2.,
                                                1.,
                                                delay_func=lambda dist: 1+(dir_delay*(dist)), 
                                                weight=sub_direction_weight,
                                                map_width=X_RESOLUTION)
                                                                                                        
[Connections_w_on,_],[Connections_w_off,_]=direction_connection_angle("W", \
                                                angle,
                                                dist, 
                                                X_RESOLUTION, 1,
                                                row_col_to_input_subsamp,                  
                                                0,1,
                                                2.,
                                                1.,
                                                delay_func=lambda dist: 1+(dir_delay*(dist)), 
                                                weight=sub_direction_weight,
                                                map_width=X_RESOLUTION)                                                                         

[Connections_n_on,_],[Connections_n_off,_]=direction_connection_angle("N", \
                                                angle,
                                                dist, 
                                                1,  Y_RESOLUTION-1, 
				                row_col_to_input_subsamp,
						0,1,
                                                2.,
                                                1.,
                                                delay_func=lambda dist: 1+(dir_delay*(dist)), 
                                                weight=sub_direction_weight,
                                                map_width=1)     
                                                
[Connections_s_on,_],[Connections_s_off,_]=direction_connection_angle("S", \
                                                angle,
                                                dist, 
                                                1,  Y_RESOLUTION-1, 
                                                row_col_to_input_subsamp, 
						0,1,    
                                                2.,
                                                1.,
                                                delay_func=lambda dist: 1+(dir_delay*(dist)), 
                                                weight=sub_direction_weight,
                                                map_width=1)                                                     
                                                
#generate paddle to inline connections
paddle_inline_east_connections=[]
paddle_inline_west_connections=[]
for i in range(X_RESOLUTION):
    for j in range(i+1, X_RESOLUTION):
        paddle_inline_west_connections.append((j, i,inline_paddle_weight, 1.))
        paddle_inline_east_connections.append((X_RESOLUTION-1-j,X_RESOLUTION-1-i, inline_paddle_weight, 1.))

#generate key_input connections
keyright_connections=[(0, 0, 10, 1.)]
keyleft_connections=[(0, 1, 10, 1.)]

#Setup projections

#key spike injector to breakout population
sim.Projection(key_input, breakout_pop, sim.OneToOneConnector(weights=5))

#breakout to subsample populations
projectionHorzSub_on=sim.Projection(breakout_pop,horz_subsamp_on_pop,sim.FromListConnector(Connections_horz_subsamp_on))
projectionSub_off=sim.Projection(breakout_pop,horz_subsamp_off_pop,sim.FromListConnector(Connections_horz_subsamp_off))

projectionHorzVertSub_on=sim.Projection(breakout_pop,vert_subsamp_on_pop,sim.FromListConnector(Connections_vert_subsamp_on))
projectionVertSub_off=sim.Projection(breakout_pop,vert_subsamp_off_pop,sim.FromListConnector(Connections_vert_subsamp_off))

#breakout to paddle populations
projectionPaddle_on=sim.Projection(breakout_pop,paddle_on_pop,sim.FromListConnector(Connections_paddle_on))
projectionPaddle_off=sim.Projection(breakout_pop,paddle_off_pop,sim.FromListConnector(Connections_paddle_off))

#paddle on population to rate_generator
sim.Projection(paddle_on_pop,rate_generator, sim.OneToOneConnector(rate_weight, 1.), target='excitatory')
#paddle off population to rate generator
sim.Projection(paddle_off_pop,rate_generator, sim.OneToOneConnector(rate_weight, 1.), target='inhibitory')

#subsample to direction populations
projectionE_on=sim.Projection(horz_subsamp_on_pop,e_on_pop,sim.FromListConnector(Connections_e_on))
projectionE_off=sim.Projection(horz_subsamp_off_pop,e_off_pop,sim.FromListConnector(Connections_e_off))
projectionW_on=sim.Projection(horz_subsamp_on_pop,w_on_pop,sim.FromListConnector(Connections_w_on))
projectionW_off=sim.Projection(horz_subsamp_off_pop,w_off_pop,sim.FromListConnector(Connections_w_off))

projectionN_on=sim.Projection(vert_subsamp_on_pop,n_on_pop,sim.FromListConnector(Connections_n_on))
projectionN_off=sim.Projection(vert_subsamp_off_pop,n_off_pop,sim.FromListConnector(Connections_n_off))
projectionS_on=sim.Projection(vert_subsamp_on_pop,s_on_pop,sim.FromListConnector(Connections_s_on))
projectionS_off=sim.Projection(vert_subsamp_off_pop,s_off_pop,sim.FromListConnector(Connections_s_off))

#rate generator (constant paddle) to inline east and west populations
sim.Projection(rate_generator,inline_east_pop,sim.FromListConnector(paddle_inline_east_connections))
sim.Projection(rate_generator,inline_west_pop,sim.FromListConnector(paddle_inline_west_connections))

#direction to inline east and west populations
sim.Projection(e_on_pop,inline_east_pop, sim.OneToOneConnector(weights=inline_direction_weight))
sim.Projection(w_on_pop,inline_west_pop, sim.OneToOneConnector(weights=inline_direction_weight))

#inhibitory north to inline connections
Connections_n_inh=[]
for i in range(Y_RESOLUTION-1):
    for j in range(X_RESOLUTION):
        inh_weight=inline_direction_weight*-1        
        Connections_n_inh.append((i,j,inh_weight,1.))   
        
#sim.Projection(n_on_pop,inline_east_pop, sim.FromListConnector(Connections_n_inh), target='inhibitory')
#sim.Projection(n_on_pop,inline_west_pop, sim.FromListConnector(Connections_n_inh), target='inhibitory')

#inline east to key input right population 
sim.Projection(inline_east_pop, key_input_right, sim.AllToAllConnector(weights=10.))
#inline west to key input left population
sim.Projection(inline_west_pop, key_input_left, sim.AllToAllConnector(weights=10.))      

#key input right and left to key input
sim.Projection(key_input_right, key_input,sim.FromListConnector(keyright_connections))
sim.Projection(key_input_left, key_input,sim.FromListConnector(keyleft_connections))

#Create visualiser
visualiser_full = spinn_breakout.Visualiser(
    breakout_port, None,
    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
    x_bits=X_BITS, y_bits=Y_BITS)
   
# Run simulation (non-blocking)
sim.run(None)

# Show visualiser (blocking)
visualiser_full.show()

# End simulation
sim.end()
