import spynnaker7.pyNN as p
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator

import pylab
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
import spinn_breakout
import sys, os
import time
import socket
import numpy as np
import math

from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork

def get_scores(breakout_pop,simulator):
    b_vertex = breakout_pop._vertex
    scores = b_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)

    return scores.tolist()

def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1, colour_bits=1, row_start=0):
    row_bits = np.uint32(row_bits)
    idx = np.uint32(0)

    if is_on_input:
        idx = 1

    row += row_start
    idx = idx | (row << (colour_bits))  # colour bit
    idx = idx | (col << (row_bits + colour_bits))

    # add two to allow for special event bits
    idx = idx + 2

    return idx


def subsample_connection(x_res, y_res, subsamp_factor_x, subsamp_factor_y, weight,
                         coord_map_func):
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on = []
    connection_list_off = []

    sx_res = int(x_res) // int(subsamp_factor_x)
    row_bits = 8#int(np.ceil(np.log2(y_res)))
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i // subsamp_factor_x
            sj = j // subsamp_factor_y
            # ON channels
            subsampidx = sj * sx_res + si
            connection_list_on.append((coord_map_func(j, i, 1, row_bits),
                                       subsampidx, weight, 1.))
            # OFF channels only on segment borders
            # if((j+1)%(y_res/subsamp_factor)==0 or (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off

def cm_to_fromlist(number_of_nodes, cm):
    i2i = []
    i2h = []
    i2o = []
    h2i = []
    h2h = []
    h2o = []
    o2i = []
    o2h = []
    o2o = []
    hidden_size = number_of_nodes - output_size - input_size
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            connect_weight = cm[j][i]
            if connect_weight != 0 and not math.isnan(connect_weight):
                if i < input_size:
                    if j < input_size:
                        i2i.append((j, i, connect_weight, delay))
                    elif j < input_size + hidden_size:
                        i2h.append((j, i, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        i2o.append((j, i, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + hidden_size:
                    if j < input_size:
                        h2i.append((j, i, connect_weight, delay))
                    elif j < input_size + hidden_size:
                        h2h.append((j, i, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        h2o.append((j, i, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + hidden_size + output_size:
                    if j < input_size:
                        o2i.append((j, i, connect_weight, delay))
                    elif j < input_size + hidden_size:
                        o2h.append((j, i, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        o2o.append((j, i, connect_weight, delay))
                    else:
                        print "shit is broke"
                else:
                    print "shit is broke"

    return i2i, i2h, i2o, h2i, h2h, h2o, o2i, o2h, o2o

def test_pop(pop):
    #test the whole population and return scores

    #Acquire all connection matrices and node types
    networks = []
    for individual in pop:
        networks.append(NeuralNetwork(individual))

    number_of_nodes = len(networks[0].cm[0])
    hidden_size = number_of_nodes - output_size - input_size

    receive_pop_size = input_size
    breakout_pops = []
    receive_on_pops = []
    hidden_node_pops = []
    output_pops = []
    weight = 0.1
    [Connections_on, Connections_off] = subsample_connection(X_RESOLUTION, Y_RESOLUTION, x_factor, y_factor, weight,
                                                             row_col_to_input_breakout)

    # Setup pyNN simulation
    p.setup(timestep=1.0)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

    #create the SpiNN nets
    for i in range(len(networks)):

        [i2i, i2h, i2o, h2i, h2h, h2o, o2i, o2h, o2o] = cm_to_fromlist(number_of_nodes, networks[i].cm)

        # Create breakout population
        breakout_pops.append(p.Population(1, spinn_breakout.Breakout, {}, label="breakout"))

        # Create input population and connect break out to it
        receive_on_pops.append(p.Population(receive_pop_size, p.IF_cond_exp, {}, label="receive_pop"))
        p.Projection(breakout_pops[i], receive_on_pops[i], p.FromListConnector(Connections_on))

        # Create output population and remaining population
        output_pops.append(p.Population(output_size, p.IF_cond_exp, {}, label="output_pop"))
        hidden_node_pops.append(p.Population(hidden_size, p.IF_cond_exp, {}, label="hidden_pop"))

        # Create the remaining nodes from the connection matrix and add them up
        if len(i2i) != 0:
            p.Projection(receive_on_pops[i], receive_on_pops[i], p.FromListConnector(i2i))
        if len(i2h) != 0:
            p.Projection(receive_on_pops[i], hidden_node_pops[i], p.FromListConnector(i2h))
        if len(i2o) != 0:
            p.Projection(receive_on_pops[i], output_pops[i], p.FromListConnector(i2o))
        if len(h2i) != 0:
            p.Projection(hidden_node_pops[i], receive_on_pops[i], p.FromListConnector(h2i))
        if len(h2h) != 0:
            p.Projection(hidden_node_pops[i], hidden_node_pops[i], p.FromListConnector(h2h))
        if len(h2o) != 0:
            p.Projection(hidden_node_pops[i], output_pops[i], p.FromListConnector(h2o))
        if len(o2i) != 0:
            p.Projection(output_pops[i], receive_on_pops[i], p.FromListConnector(o2i))
        if len(o2h) != 0:
            p.Projection(output_pops[i], hidden_node_pops[i], p.FromListConnector(o2h))
        if len(o2o) != 0:
            p.Projection(output_pops[i], output_pops[i], p.FromListConnector(o2o))



    print "reached here 1"

    runtime = 31000

    simulator = get_simulator()

    p.run(runtime)

    print "reached here 2"

    scores = []
    for i in range(len(networks)):
        scores.append(get_scores(breakout_pop=breakout_pops[i], simulator=simulator))

    # End simulation
    p.end()

    print scores

X_BITS = 8
Y_BITS = 8

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT1 = 17887
UDP_PORT2 = UDP_PORT1 + 1

weight_max = 0.03
delay = 5

x_res = 160
y_res = 128
x_factor = 4
y_factor = 4

input_size = (x_res/x_factor)*(y_res/y_factor)
output_size = 2

genotype = lambda: NEATGenotype(inputs=input_size,
                                outputs=output_size,
                                weight_range=(-50, 50),
                                types=['excitatory', 'inhibitory'],
                                feedforward=False)

# Create a population
pop = NEATPopulation(genotype, popsize=20)

# Run the evolution, tell it to use the task as an evaluator
pop.epoch(generations=200, evaluator=test_pop, solution=None, SpiNNaker=True)


