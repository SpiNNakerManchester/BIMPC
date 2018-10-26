import spynnaker7.pyNN as p
import time
import gc
import sys
import copy
import pylab
import numpy as np
import threading
from threading import Condition
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pyNN.random import RandomDistribution# as rand
#import spynnaker8.spynakker_plotting as splot
from pympler.tracker import SummaryTracker
import spinn_breakout

input_size = 160*128
hidden_size = 100
output_size = 100

def function(tracker):
    p.setup(timestep=1.0)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

    i2hcon = []
    for i in range(input_size):
        for j in range(hidden_size):
            i2hcon.append((i, j, 0.5, 1.))

    input_pop = p.Population(input_size, p.IF_cond_exp, {}, label='input')
    hidden_pop = p.Population(hidden_size, p.IF_cond_exp, {},  label='hidden')
    output_pop = p.Population(output_size, p.IF_cond_exp, {},  label='output')
    breakout = p.Population(1, spinn_breakout.Breakout, {}, label="breakout")
    print "after populations"
    tracker.print_diff()
    conn = p.FromListConnector(i2hcon, tracker=tracker)
    print "size = ", sys.getsizeof(conn)
    print "after conn"
    tracker.print_diff()
    a = p.Projection(input_pop, hidden_pop, p.AllToAllConnector(weights=0.5, delays=1.))
    print "after i->h"
    tracker.print_diff()
    a1 = p.Projection(input_pop, hidden_pop, conn)
    print "after i->h from list"
    tracker.print_diff()
    # a1 = p.Projection(input_pop, hidden_pop, p.FromListConnector(i2hcon, tracker))
    # print "after i->h from list 2"
    # tracker.print_diff()
    # a2 = p.Projection(breakout, hidden_pop, p.AllToAllConnector(weights=0.5, delays=1.))
    # print "after b->h"
    # tracker.print_diff()
    # b = p.Projection(input_pop, output_pop, p.AllToAllConnector(weights=2, delays=1.))
    # print "after i->o"
    # tracker.print_diff()
    # c = p.Projection(hidden_pop, output_pop, p.AllToAllConnector(weights=2, delays=1.))
    # print "after h->o"
    # tracker.print_diff()
    # d = p.Projection(input_pop, input_pop, p.AllToAllConnector(weights=2, delays=1.))
    # print "after i->i"
    # tracker.print_diff()
    # a = p.Projection(input_pop, hidden_pop, p.AllToAllConnector(weights=2, delays=1.))
    # print "after i->h 2"
    # tracker.print_diff()
    # a2 = p.Projection(breakout, hidden_pop, p.AllToAllConnector(weights=0.5, delays=1.))
    # print "after b->h 2"
    # tracker.print_diff()
    # a1 = p.Projection(input_pop, hidden_pop, p.AllToAllConnector(weights=0.5, delays=1.))
    # print "after i->h a2a overwrite"
    # tracker.print_diff()
    a1 = None
    print "after a1 empty"
    tracker.print_diff()

    p.end()
    print "after end"
    tracker.print_diff()

    del conn
    print "after del"
    tracker.print_diff()
    conn = None
    print "after none"
    tracker.print_diff()


    gc.collect()
    print "after collect test"
    tracker.print_diff()


tracker = SummaryTracker()
tracker.print_diff()

function(tracker)
print "exited function"
tracker.print_diff()

print "finished"

# p.Projection(breakout_pops[i], receive_on_pops[i], p.FromListConnector(Connections_on))

