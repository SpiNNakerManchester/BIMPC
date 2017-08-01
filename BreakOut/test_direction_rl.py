from pacman.model.constraints.partitioner_constraints.partitioner_maximum_size_constraint import \
    PartitionerMaximumSizeConstraint

import spynnaker7.pyNN as sim
from breakout_utils import get_punishment_neuron_id, get_reward_neuron_id
from spynnaker_external_devices_plugin.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker_external_devices_plugin.pyNN.spynnaker_external_device_plugin_manager import SpynnakerExternalDevicePluginManager as ex
import spynnaker_external_devices_plugin.pyNN as ex
import spinn_controller
import spinn_breakout

# Layout of pixels
from spynnaker_external_devices_plugin.pyNN.utility_models.spike_injector import SpikeInjector
from spinn_breakout.visualiser.visualiser import Visualiser
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

paddle_row = Y_RESOLUTION - 1

# paddle size
paddle_width = 8
# pixel speed
pps = 50.  # speed for 20ms frame rate in bkout.c
# movement timestep
pt = 1. / pps
# number of pixels to calculate speed across
div = 2

# UDP port to read spikes from
breakout_port = 17893

# Setup pyNN simulation
timestep = 1.
sim.setup(timestep=timestep)

# restrict number of neurons on each core
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

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

# rate generator parameters
rate_weight = 1.5
rate_delay = 16.

### Hold:
###controller_sink = sim.Population(1, sim.IF_curr_exp, {}, label="sink")
###sim.Projection(controller_pop, controller_sink, sim.OneToOneConnector(1), label='sink connection')

# Create breakout population and activate live output for it
breakout_pop = sim.Population(1, spinn_breakout.Breakout, {}, label="breakout")
controller_pop = sim.Population(1, spinn_controller.RLController, {}, label="controller")
sim.Projection(controller_pop, breakout_pop, sim.FromListConnector([(0,0,1,1),(1,1,1,1)]), label='controlOut')
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=breakout_port)
# Create key input population
paddle_controller = sim.Population(2, sim.IF_curr_exp_supervision, cell_params_lif, label="key_input")
# Create spike injector to inject keyboard input into simulation
key_input = sim.Population(2, SpikeInjector, {"port": 12367}, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])
# Connect key spike injector to breakout population
sim.Projection(key_input, paddle_controller, sim.OneToOneConnector(weights=5))
# Create subsampled pop
breakout_size = 2 ** (X_BITS + Y_BITS + 1) + 2
# print "breakout population size: ", breakout_size
# print "subsampled (factor of", subsamp_factor,") population size: ", subsamp_size
horz_subsamp_on_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="subsample channel on")
horz_subsamp_off_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="subsample channel off")
vert_subsamp_on_pop = sim.Population(Y_RESOLUTION - 1, sim.IF_curr_exp, {}, label="vertical subsample channel on")
vert_subsamp_off_pop = sim.Population(Y_RESOLUTION - 1, sim.IF_curr_exp, {}, label="vertical subsample channel off")

# Create paddle detection populations
paddle_on_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="paddle channel on")
paddle_off_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="paddle channel off")

# create rate generator population (for constant paddle output)
rate_generator = sim.Population(X_RESOLUTION, sim.IF_curr_exp, cell_params_lif,
                                label="Rate generation population")
sim.Projection(rate_generator, rate_generator, sim.OneToOneConnector(rate_weight, rate_delay), target='excitatory',
               label='rate_pop->rate_pop')

# create east and west direction populations
e_on_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="e_on")
e_off_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="e_off")
w_on_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="w_on")
w_off_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="w_off")

n_on_pop = sim.Population(Y_RESOLUTION - 1, sim.IF_curr_exp, {}, label="n_on")
n_off_pop = sim.Population(Y_RESOLUTION - 1, sim.IF_curr_exp, {}, label="n_off")
s_on_pop = sim.Population(Y_RESOLUTION - 1, sim.IF_curr_exp, {}, label="s_on")
s_off_pop = sim.Population(Y_RESOLUTION - 1, sim.IF_curr_exp, {}, label="s_off")

# create inline population
inline_east_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="inline east channel")
inline_west_pop = sim.Population(X_RESOLUTION, sim.IF_curr_exp, {}, label="inline west channel")

# Create key input right and left populations
key_input_right = sim.Population(1, sim.IF_curr_exp, cell_params_lif, label="key_input_right")
key_input_left = sim.Population(1, sim.IF_curr_exp, cell_params_lif, label="key_input_left")

# declare projection weights
# breakout--> subsample
subweight = 5  # 2.7
# subsample-->direction connections
sub_direction_weight = 2.5
inline_direction_weight = 2.7  # 3.34
inline_paddle_weight = 1. / paddle_width  # inline_direction_weight/paddle_width

# generate breakout to subsample connection lists (note Y-RESOLUTION-1 to ignore paddle row)
[Connections_horz_subsamp_on, Connections_horz_subsamp_off] = subsample_connection(X_RESOLUTION, Y_RESOLUTION - 1, 1,
                                                                                   Y_RESOLUTION - 1, \
                                                                                   subweight, row_col_to_input_breakout)

[Connections_vert_subsamp_on, Connections_vert_subsamp_off] = subsample_connection(X_RESOLUTION, Y_RESOLUTION - 1,
                                                                                   X_RESOLUTION, 1, \
                                                                                   subweight, row_col_to_input_breakout)

# generate breakout to paddle connection lists
[Connections_paddle_on, Connections_paddle_off] = paddle_connection(X_RESOLUTION, paddle_row, 1, subweight,
                                                                    row_col_to_input_breakout)

# generate subsample to direction connection lists
dist = 1
angle = 8
dir_delay = int(pt * 1000)

# TODO: make these non-uniform distributions, larger weights in the centre of image?
[Connections_e_on, _], [Connections_e_off, _] = direction_connection_angle("E", \
                                                                           angle,
                                                                           dist,
                                                                           X_RESOLUTION, 1,
                                                                           row_col_to_input_subsamp,
                                                                           0, 1,
                                                                           2.,
                                                                           1.,
                                                                           delay_func=lambda dist: 1 + (
                                                                               dir_delay * (dist)),
                                                                           weight=sub_direction_weight,
                                                                           map_width=X_RESOLUTION)

[Connections_w_on, _], [Connections_w_off, _] = direction_connection_angle("W", \
                                                                           angle,
                                                                           dist,
                                                                           X_RESOLUTION, 1,
                                                                           row_col_to_input_subsamp,
                                                                           0, 1,
                                                                           2.,
                                                                           1.,
                                                                           delay_func=lambda dist: 1 + (
                                                                               dir_delay * (dist)),
                                                                           weight=sub_direction_weight,
                                                                           map_width=X_RESOLUTION)

[Connections_n_on, _], [Connections_n_off, _] = direction_connection_angle("N", \
                                                                           angle,
                                                                           dist,
                                                                           1, Y_RESOLUTION - 1,
                                                                           row_col_to_input_subsamp,
                                                                           0, 1,
                                                                           2.,
                                                                           1.,
                                                                           delay_func=lambda dist: 1 + (
                                                                               dir_delay * (dist)),
                                                                           weight=sub_direction_weight,
                                                                           map_width=1)

[Connections_s_on, _], [Connections_s_off, _] = direction_connection_angle("S", \
                                                                           angle,
                                                                           dist,
                                                                           1, Y_RESOLUTION - 1,
                                                                           row_col_to_input_subsamp,
                                                                           0, 1,
                                                                           2.,
                                                                           1.,
                                                                           delay_func=lambda dist: 1 + (
                                                                               dir_delay * (dist)),
                                                                           weight=sub_direction_weight,
                                                                           map_width=1)

# generate paddle to inline connections
paddle_inline_east_connections = []
paddle_inline_west_connections = []
for i in range(X_RESOLUTION):
    for j in range(i + 1, X_RESOLUTION):
        paddle_inline_west_connections.append((j, i, inline_paddle_weight, 1.))
        paddle_inline_east_connections.append((X_RESOLUTION - 1 - j, X_RESOLUTION - 1 - i, inline_paddle_weight, 1.))

# generate key_input connections
keyright_connections = [(0, 0, 5, 1.)]
keyleft_connections = [(0, 1, 5, 1.)]

# Setup projections

# key spike injector to breakout population
sim.Projection(paddle_controller, breakout_pop, sim.OneToOneConnector(weights=5))

# breakout to subsample populations
projectionHorzSub_on = sim.Projection(breakout_pop, horz_subsamp_on_pop,
                                      sim.FromListConnector(Connections_horz_subsamp_on))
projectionSub_off = sim.Projection(breakout_pop, horz_subsamp_off_pop,
                                   sim.FromListConnector(Connections_horz_subsamp_off))

projectionHorzVertSub_on = sim.Projection(breakout_pop, vert_subsamp_on_pop,
                                          sim.FromListConnector(Connections_vert_subsamp_on))
projectionVertSub_off = sim.Projection(breakout_pop, vert_subsamp_off_pop,
                                       sim.FromListConnector(Connections_vert_subsamp_off))

# breakout to paddle populations
projectionPaddle_on = sim.Projection(breakout_pop, paddle_on_pop, sim.FromListConnector(Connections_paddle_on))
projectionPaddle_off = sim.Projection(breakout_pop, paddle_off_pop, sim.FromListConnector(Connections_paddle_off))

# paddle on population to rate_generator
sim.Projection(paddle_on_pop, rate_generator, sim.OneToOneConnector(rate_weight, 1.), target='excitatory')
# paddle off population to rate generator
sim.Projection(paddle_off_pop, rate_generator, sim.OneToOneConnector(rate_weight, 1.), target='inhibitory')

# subsample to direction populations
projectionE_on = sim.Projection(horz_subsamp_on_pop, e_on_pop, sim.FromListConnector(Connections_e_on))
projectionE_off = sim.Projection(horz_subsamp_off_pop, e_off_pop, sim.FromListConnector(Connections_e_off))
projectionW_on = sim.Projection(horz_subsamp_on_pop, w_on_pop, sim.FromListConnector(Connections_w_on))
projectionW_off = sim.Projection(horz_subsamp_off_pop, w_off_pop, sim.FromListConnector(Connections_w_off))

projectionN_on = sim.Projection(vert_subsamp_on_pop, n_on_pop, sim.FromListConnector(Connections_n_on))
projectionN_off = sim.Projection(vert_subsamp_off_pop, n_off_pop, sim.FromListConnector(Connections_n_off))
projectionS_on = sim.Projection(vert_subsamp_on_pop, s_on_pop, sim.FromListConnector(Connections_s_on))
projectionS_off = sim.Projection(vert_subsamp_off_pop, s_off_pop, sim.FromListConnector(Connections_s_off))

# rate generator (constant paddle) to inline east and west populations
sim.Projection(rate_generator, inline_east_pop, sim.FromListConnector(paddle_inline_east_connections))
sim.Projection(rate_generator, inline_west_pop, sim.FromListConnector(paddle_inline_west_connections))

# direction to inline east and west populations
sim.Projection(e_on_pop, inline_east_pop, sim.OneToOneConnector(weights=inline_direction_weight))
sim.Projection(w_on_pop, inline_west_pop, sim.OneToOneConnector(weights=inline_direction_weight))

# inhibitory north to inline connections
Connections_n_inh = []
for i in range(Y_RESOLUTION - 1):
    for j in range(X_RESOLUTION):
        inh_weight = inline_direction_weight * -1
        Connections_n_inh.append((i, j, inh_weight, 1.))

        # sim.Projection(n_on_pop,inline_east_pop, sim.FromListConnector(Connections_n_inh), target='inhibitory')
# sim.Projection(n_on_pop,inline_west_pop, sim.FromListConnector(Connections_n_inh), target='inhibitory')

# RL part
# Setup Actor and Critic populations
population_size = 500
connection_probability = .1
# State representation
state_population = sim.Population(population_size, sim.IF_curr_exp,
                                  cellparams=cell_params_lif, label="state pop")
actor_population = sim.Population(population_size, sim.IF_curr_exp_supervision,
                                  cellparams=pyramidal_lif, label="state pop")

sim.Projection(state_population, state_population,
               sim.FixedProbabilityConnector(p_connect=connection_probability * 2, weights=.3)
               )

state_population.set_constraint(PartitionerMaximumSizeConstraint(100))
actor_population.set_constraint(PartitionerMaximumSizeConstraint(15))

reward_pop = sim.Population(1, sim.IF_curr_exp, cell_params_lif,
                            label="reward_pop")
punishment_pop = sim.Population(1, sim.IF_curr_exp, cell_params_lif,
                                label="punishment_pop")
sim.Projection(breakout_pop, reward_pop, sim.FromListConnector([(get_reward_neuron_id(), 0, 2, 1)]))
sim.Projection(breakout_pop, punishment_pop, sim.FromListConnector([(get_punishment_neuron_id(), 0, 2, 2)]))

# Reward
sim.Projection(reward_pop, actor_population, sim.AllToAllConnector(weights=.1, delays=1),
               target="reward", label='reward -> actor_l')
# sim.Projection(reward_pop, key_input_left, sim.AllToAllConnector(weights=.001, delays=1),
#                target="reward", label='reward -> actor_r')

# Punishment

# sim.Projection(punishment_pop, key_input_right, sim.AllToAllConnector(weights=.0005, delays=1),
#                target="punishment", label='reward -> actor_l')
# sim.Projection(punishment_pop, key_input_left, sim.AllToAllConnector(weights=.0005, delays=1),
#                target="punishment", label='reward -> actor_r')
# Supervision (TD error) signal connection
synapse_dynamics = sim.SynapseDynamics(slow=sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=15.0, tau_minus=30.0, tau_c=2.0, tau_d=200.0),
    # Eligibility trace and dopamine constants
    weight_dependence=sim.AdditiveWeightDependence(), mad=True,
    neuromodulation=True))

# Whatever
sim.Projection(state_population, actor_population,
               sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )

# inline east to key input right population
sim.Projection(e_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# inline west to key input left population
sim.Projection(e_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# inline east to key input right population
sim.Projection(w_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# inline west to key input left population
sim.Projection(w_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# Rate
sim.Projection(rate_generator, state_population,
               sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# inline west to key input left population
sim.Projection(rate_generator, state_population,
               sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )

# inline east to key input right population
sim.Projection(n_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# inline west to key input left population
sim.Projection(n_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )

# inline east to key input right population
sim.Projection(s_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# inline west to key input left population
sim.Projection(s_on_pop, state_population, sim.FixedProbabilityConnector(p_connect=connection_probability, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# learned part
sim.Projection(actor_population, key_input_left, sim.FixedNumberPreConnector(population_size // 2, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
sim.Projection(actor_population, key_input_right, sim.FixedNumberPreConnector(population_size // 2, weights=.2),
               # synapse_dynamics=synapse_dynamics
               )
# key input right and left to key input
sim.Projection(key_input_right, paddle_controller, sim.FromListConnector(keyright_connections),
               synapse_dynamics=synapse_dynamics
               )
sim.Projection(key_input_left, paddle_controller, sim.FromListConnector(keyleft_connections),
               synapse_dynamics=synapse_dynamics
               )

poisson_noise = sim.Population(50, sim.SpikeSourcePoisson, {'rate': 10.})
# poisson_noise_r = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 10.})

# sim.Projection(poisson_noise_l, key_input_right, sim.AllToAllConnector(weights=.4, delays=1),
#                target="excitatory", label='poisson -> actor')
# sim.Projection(poisson_noise_r, key_input_left, sim.AllToAllConnector(weights=.4, delays=1),
#                target="excitatory", label='poisson -> actor')

state_poisson_noise = sim.Population(50, sim.SpikeSourcePoisson, {'rate': 10.})
sim.Projection(state_poisson_noise, state_population,
               sim.FixedProbabilityConnector(connection_probability * 4, weights=.3, delays=1),
               target="excitatory", label='poisson -> actor')
sim.Projection(poisson_noise, actor_population,
               sim.FixedProbabilityConnector(connection_probability * 4, weights=.3, delays=1),
               target="excitatory", label='poisson -> actor')
# Create visualiser
visualiser_full = Visualiser(
    breakout_port, key_input_connection,
    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
    x_bits=X_BITS, y_bits=Y_BITS)

# Run simulation (non-blocking)
sim.run(None)

# Show visualiser (blocking)
visualiser_full.show()

# End simulation
sim.end()
