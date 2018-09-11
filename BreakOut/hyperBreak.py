import spynnaker7.pyNN as p
from functools import partial
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator
from pympler.tracker import SummaryTracker

import pylab
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
import spinn_breakout
import sys, os
import time
import socket
import numpy as np
import csv
import math

from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork
from peas.methods.evolution import SimplePopulation
from peas.methods.wavelets import WaveletGenotype, WaveletDeveloper
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.tasks.linefollowing import LineFollowingTask

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
    i2i_ex = []
    i2i_in = []
    i2h_ex = []
    i2h_in = []
    i2o_ex = []
    i2o_in = []
    h2i_ex = []
    h2i_in = []
    h2h_ex = []
    h2h_in = []
    h2o_ex = []
    h2o_in = []
    o2i_ex = []
    o2i_in = []
    o2h_ex = []
    o2h_in = []
    o2o_ex = []
    o2o_in = []
    hidden_size = number_of_nodes - output_size - input_size
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            connect_weight = cm[j][i] * (weight_max / 50.)
            if connect_weight != 0 and not math.isnan(connect_weight):
                if i < input_size:
                    if j < input_size:
                        if connect_weight > 0:
                            i2i_ex.append((i, j, connect_weight, delay))
                        else:
                            i2i_in.append((i, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        if connect_weight > 0:
                            i2o_ex.append((i, j-input_size, connect_weight, delay))
                        else:
                            i2o_in.append((i, j - input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        if connect_weight > 0:
                            i2h_ex.append((i, j-input_size-output_size, connect_weight, delay))
                        else:
                            i2h_in.append((i, j-input_size-output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + output_size:
                    if j < input_size:
                        if connect_weight > 0:
                            o2i_ex.append((i-input_size, j, connect_weight, delay))
                        else:
                            o2i_in.append((i-input_size, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        if connect_weight > 0:
                            o2o_ex.append((i-input_size, j-input_size, connect_weight, delay))
                        else:
                            o2o_in.append((i-input_size, j-input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        if connect_weight > 0:
                            o2h_ex.append((i-input_size, j-input_size-output_size, connect_weight, delay))
                        else:
                            o2h_in.append((i-input_size, j-input_size-output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                elif i < input_size + hidden_size + output_size:
                    if j < input_size:
                        if connect_weight > 0:
                            h2i_ex.append((i-input_size-output_size, j, connect_weight, delay))
                        else:
                            h2i_in.append((i-input_size-output_size, j, connect_weight, delay))
                    elif j < input_size + output_size:
                        if connect_weight > 0:
                            h2o_ex.append((i-input_size-output_size, j-input_size, connect_weight, delay))
                        else:
                            h2o_in.append((i-input_size-output_size, j-input_size, connect_weight, delay))
                    elif j < input_size + hidden_size + output_size:
                        if connect_weight > 0:
                            h2h_ex.append((i-input_size-output_size, j-input_size-output_size, connect_weight, delay))
                        else:
                            h2h_in.append((i-input_size-output_size, j-input_size-output_size, connect_weight, delay))
                    else:
                        print "shit is broke"
                else:
                    print "shit is broke"

    return i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in, h2o_in, o2i_in, o2h_in, o2o_in

def connect_genes_to_fromlist(number_of_nodes, indiviudal):
    i2i = []
    i2h = []
    i2o = []
    h2i = []
    h2h = []
    h2o = []
    o2i = []
    o2h = []
    o2o = []

    #individual: Tuples of (innov, from, to, weight, enabled)

    hidden_size = number_of_nodes - output_size - input_size

    for connections in indiviudal:
        c = indiviudal[connections]
        connect_weight = c[3] * (weight_max / 50.)
        if c[4] == True:
            if c[1] < input_size:
                if c[2] < input_size:
                    i2i.append((c[1], c[2], connect_weight, delay))
                elif c[2] < input_size + output_size:
                    i2o.append((c[1], c[2]-input_size, connect_weight, delay))
                elif c[2] < input_size + hidden_size + output_size:
                    i2h.append((c[1], c[2]-input_size-output_size, connect_weight, delay))
                else:
                    print "shit broke"
            elif c[1] < input_size + output_size:
                if c[2] < input_size:
                    o2i.append((c[1]-input_size, c[2], connect_weight, delay))
                elif c[2] < input_size + output_size:
                    o2o.append((c[1]-input_size, c[2]-input_size, connect_weight, delay))
                elif c[2] < input_size + hidden_size + output_size:
                    o2h.append((c[1]-input_size, c[2]-input_size-output_size, connect_weight, delay))
                else:
                    print "shit broke"
            elif c[1] < input_size + hidden_size + output_size:
                if c[2] < input_size:
                    h2i.append((c[1]-input_size-output_size, c[2], connect_weight, delay))
                elif c[2] < input_size + output_size:
                    h2o.append((c[1]-input_size-output_size, c[2]-input_size, connect_weight, delay))
                elif c[2] < input_size + hidden_size + output_size:
                    h2h.append((c[1]-input_size-output_size, c[2]-input_size-output_size, connect_weight, delay))
                else:
                    print "shit broke"
            else:
                print "shit broke"


    return i2i, i2h, i2o, h2i, h2h, h2o, o2i, o2h, o2o

def test_pop(pop, tracker):
    #test the whole population and return scores

    #Acquire all connection matrices and node types
    # networks = []
    # for individual in pop:
    #     networks.append(NeuralNetwork(individual))

    receive_pop_size = input_size
    breakout_pops = []
    receive_on_pops = []
    hidden_node_pops = []
    hidden_count = 0
    output_pops = []
    # weight = 0.1
    # [Connections_on, Connections_off] = subsample_connection(X_RESOLUTION, Y_RESOLUTION, x_factor, y_factor, weight,
    #                                                          row_col_to_input_breakout)

    # Setup pyNN simulation
    p.setup(timestep=1.0)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

    print len(pop)
    #create the SpiNN nets
    for i in range(len(pop)):
        print "converting geno->pheno->spino ", i
        phenotype = developer.convert(pop[i])

        number_of_nodes = len(phenotype.node_types)
        hidden_size = number_of_nodes - output_size - input_size

        # [i2i, i2h, i2o, h2i, h2h, h2o, o2i, o2h, o2o] = connect_genes_to_fromlist(number_of_nodes, phenotype.conn_genes)

        i2i_ex, i2h_ex, i2o_ex, h2i_ex, h2h_ex, h2o_ex, o2i_ex, o2h_ex, o2o_ex, i2i_in, i2h_in, i2o_in, h2i_in, h2h_in, \
        h2o_in, o2i_in, o2h_in, o2o_in = cm_to_fromlist(number_of_nodes, phenotype.cm)

        # Create breakout population
        breakout_pops.append(p.Population(1, spinn_breakout.Breakout, {}, label="breakout {}".format(i)))

        # Create input population and connect break out to it
        receive_on_pops.append(p.Population(receive_pop_size, p.IF_cond_exp, {}, label="receive_pop {}".format(i)))
        p.Projection(breakout_pops[i], receive_on_pops[i], breakout_connections)

        # Create output population and remaining population
        output_pops.append(p.Population(output_size, p.IF_cond_exp, {}, label="output_pop {}".format(i)))
        p.Projection(output_pops[i], breakout_pops[i], p.AllToAllConnector())

        if hidden_size != 0:
            hidden_node_pops.append(p.Population(hidden_size, p.IF_cond_exp, {}, label="hidden_pop {}".format(i)))
            hidden_count += 1
        #     hidden_node_pops[hidden_count-1].record()
        # receive_on_pops[i].record()
        # output_pops[i].record()

        # Create the remaining nodes from the connection matrix and add them up
        if len(i2i_ex) != 0:
            p.Projection(receive_on_pops[i], receive_on_pops[i], p.FromListConnector(i2i_ex), target='excitatory')
        if len(i2h_ex) != 0:
            p.Projection(receive_on_pops[i], hidden_node_pops[hidden_count-1], p.FromListConnector(i2h_ex), target='excitatory')
        if len(i2o_ex) != 0:
            p.Projection(receive_on_pops[i], output_pops[i], p.FromListConnector(i2o_ex), target='excitatory')
        if len(h2i_ex) != 0:
            p.Projection(hidden_node_pops[hidden_count-1], receive_on_pops[i], p.FromListConnector(h2i_ex), target='excitatory')
        if len(h2h_ex) != 0:
            p.Projection(hidden_node_pops[hidden_count-1], hidden_node_pops[hidden_count-1], p.FromListConnector(h2h_ex), target='excitatory')
        if len(h2o_ex) != 0:
            p.Projection(hidden_node_pops[hidden_count-1], output_pops[i], p.FromListConnector(h2o_ex), target='excitatory')
        if len(o2i_ex) != 0:
            p.Projection(output_pops[i], receive_on_pops[i], p.FromListConnector(o2i_ex), target='excitatory')
        if len(o2h_ex) != 0:
            p.Projection(output_pops[i], hidden_node_pops[hidden_count-1], p.FromListConnector(o2h_ex), target='excitatory')
        if len(o2o_ex) != 0:
            p.Projection(output_pops[i], output_pops[i], p.FromListConnector(o2o_ex), target='excitatory')
        if len(i2i_in) != 0:
            p.Projection(receive_on_pops[i], receive_on_pops[i], p.FromListConnector(i2i_in), target='inhibitory')
        if len(i2h_in) != 0:
            p.Projection(receive_on_pops[i], hidden_node_pops[hidden_count-1], p.FromListConnector(i2h_in), target='inhibitory')
        if len(i2o_in) != 0:
            p.Projection(receive_on_pops[i], output_pops[i], p.FromListConnector(i2o_in), target='inhibitory')
        if len(h2i_in) != 0:
            p.Projection(hidden_node_pops[hidden_count-1], receive_on_pops[i], p.FromListConnector(h2i_in), target='inhibitory')
        if len(h2h_in) != 0:
            p.Projection(hidden_node_pops[hidden_count-1], hidden_node_pops[hidden_count-1], p.FromListConnector(h2h_in), target='inhibitory')
        if len(h2o_in) != 0:
            p.Projection(hidden_node_pops[hidden_count-1], output_pops[i], p.FromListConnector(h2o_in), target='inhibitory')
        if len(o2i_in) != 0:
            p.Projection(output_pops[i], receive_on_pops[i], p.FromListConnector(o2i_in), target='inhibitory')
        if len(o2h_in) != 0:
            p.Projection(output_pops[i], hidden_node_pops[hidden_count-1], p.FromListConnector(o2h_in), target='inhibitory')
        if len(o2o_in) != 0:
            p.Projection(output_pops[i], output_pops[i], p.FromListConnector(o2o_in), target='inhibitory')



    print "reached here 1"

    runtime = 501000

    simulator = get_simulator()

    p.run(runtime)

    print "reached here 2"

    scores = []
    for i in range(len(pop)):
        scores.append(get_scores(breakout_pop=breakout_pops[i], simulator=simulator))
        pop[i].stats = {'fitness': scores[i][len(scores[i])-1][0]}#, 'steps': 0}

    #     if i == 0:
    #         pylab.figure()
    #     spikes_on = output_pops[i].getSpikes()
    #     ax = pylab.subplot(1, len(pop), i+1)#4, 1)
    #     pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    #     pylab.xlabel("Time (ms)")
    #     pylab.ylabel("neuron ID")
    #     pylab.axis([0, runtime, -1, output_size + 1])
    # pylab.show()
    # pylab.figure()
    # for i in range(hidden_count):
    #     spikes_on = hidden_node_pops[i].getSpikes()
    #     ax = pylab.subplot(1, len(pop), i+1)#4, 1)
    #     pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    #     pylab.xlabel("Time (ms)")
    #     pylab.ylabel("neuron ID")
    #     pylab.axis([0, runtime, -1, receive_pop_size + 1])
    # pylab.show()
    # pylab.figure()
    # for i in range(len(pop)):
    #     spikes_on = receive_on_pops[i].getSpikes()
    #     ax = pylab.subplot(1, len(pop), i+1)#4, 1)
    #     pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
    #     pylab.xlabel("Time (ms)")
    #     pylab.ylabel("neuron ID")
    #     pylab.axis([0, runtime, -1, receive_pop_size + 1])
    # pylab.show()

    j = 0
    for score in scores:
        print j, score
        j += 1

    print "factors: ", x_factor

    gen_stats(pop)
    save_champion()
    # End simulation
    p.end()

def gen_stats(list_pop):
    # pop._gather_stats(list_pop)
    for stat in pop.stats:
        print "{}: {}".format(stat, pop.stats[stat])

def save_champion():
    iteration = len(pop.champions) - 1
    if iteration >= 0:
        with open('hyper champion {} - {}.csv'.format(iteration, x_factor), 'w') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for i in pop.champions[iteration].conn_genes:
                writer.writerow(pop.champions[iteration].conn_genes[i])
            for i in pop.champions[iteration].node_genes:
                writer.writerow(i)
            for i in pop.champions[iteration].stats:
                writer.writerow(["fitness", pop.champions[iteration].stats[i]])
            # writer.writerow("\n")
        with open('hyper champions {}.csv'.format(x_factor), 'a') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for i in pop.champions[iteration].conn_genes:
                writer.writerow(pop.champions[iteration].conn_genes[i])
            for i in pop.champions[iteration].node_genes:
                writer.writerow(i)
            for i in pop.champions[iteration].stats:
                writer.writerow(["fitness", pop.champions[iteration].stats[i]])



X_BITS = 8
Y_BITS = 8

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT1 = 17887
UDP_PORT2 = UDP_PORT1 + 1

weight_max = 1
delay = 2

x_res = 160
y_res = 128
x_factor = 8
y_factor = 8

input_size = (x_res/x_factor)*(y_res/y_factor)
output_size = 2

weight = 0.1
[Connections_on, Connections_off] = subsample_connection(X_RESOLUTION, Y_RESOLUTION, x_factor, y_factor, weight,
                                                         row_col_to_input_breakout)

p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
breakout_connections = p.FromListConnector(Connections_on)
p.end()

# Configure substrate
substrate = Substrate()
print "adding input"
substrate.add_nodes([(r, theta) for r in np.linspace(-1,1,int(x_res/x_factor))
                          for theta in np.linspace(-1, 1, int(y_res/y_factor))], 'input')
print "adding output"
substrate.add_nodes([(r, theta) for r in np.linspace(0,0,1)
                          for theta in np.linspace(-1, 1, 2)], 'output')
print "adding hidden"
substrate.add_nodes([(r, theta) for r in np.linspace(-1,1,int(x_res/x_factor))
                          for theta in np.linspace(-1, 1, int(y_res/y_factor))], 'hidden')

print "adding connections"
# substrate.add_connections('input', 'input',-1)
substrate.add_connections('input', 'hidden', -1)
# substrate.add_connections('input', 'output',-1)
# substrate.add_connections('hidden', 'input',-2)
# substrate.add_connections('hidden', 'hidden', -2)
substrate.add_connections('hidden', 'output',-2)
# substrate.add_connections('output', 'input',-3)
# substrate.add_connections('output', 'hidden', -3)
# substrate.add_connections('output', 'output',-3)

print "setting up classes etc"
geno_kwds = dict(feedforward=True,
                 inputs=4,
                 outputs=2,
                 weight_range=(-50.0, 50.0),
                 prob_add_conn=0.1,
                 prob_add_node=0.06,
                 bias_as_node=False,
                 types=['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'])

geno = lambda: NEATGenotype(**geno_kwds)

pop = NEATPopulation(geno, popsize=100, target_species=8)

developer = HyperNEATDeveloper(substrate=substrate,
                               add_deltas=False,
                               sandwich=False,
                               feedforward=False,
                               node_type=['excitatory', 'inhibitory'])


# genotype = lambda: NEATGenotype(inputs=input_size,
#                                 outputs=output_size,
#                                 prob_add_node=0.7,
#                                 prob_add_conn=0.4,
#                                 weight_range=(-50, 50),
#                                 types=['excitatory', 'inhibitory'],
#                                 feedforward=False)
#
# # Create a population
# pop = NEATPopulation(genotype, popsize=100)

# Run the evolution, tell it to use the task as an evaluator
print "beginning epoch"
tracker = SummaryTracker()
results = pop.epoch(tracker=tracker, generations=2000,
                    evaluator=test_pop,
                    solution=None,
                    SpiNNaker=True
                    )

