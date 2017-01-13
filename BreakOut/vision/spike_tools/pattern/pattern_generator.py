from __future__ import print_function

import sys
import os
import time
import numpy as np
import math
from numpy import float64
from numpy import random
import pickle
import glob


LESS_THAN     = "less than"
GREATER_THAN  = "greater than"
LEFT_TO_RIGHT = "left to right"
RIGHT_TO_LEFT = "right to left"
TOP_TO_BOTTOM = "top to bottom"
BOTTOM_TO_TOP = "bottom to top"
HORZ = 0
VERT = 1


#---------- From NE15 ----------#

def nextTime(rateParameter):
    '''Helper function to Poisson generator (from NE15)
       :param rateParameter: The rate at which a neuron will fire (Hz)
       
       :returns: Time at which the neuron should spike next (seconds)
    '''
    return -math.log(1.0 - random.random()) / rateParameter


def poisson_generator(rate, t_start, t_stop):
    '''Poisson train generator (from NE15)
       :param rate: The rate at which a neuron will fire (Hz)
       :param t_start: When should the neuron start to fire (milliseconds)
       :param t_stop: When should the neuron stop firing (milliseconds)
       
       :returns: Poisson train firing at rate, from t_start to t_stop (milliseconds)
    '''
    poisson_train = []
    if rate > 0:
        next_isi = nextTime(rate)*1000.
        last_time = next_isi + t_start
        while last_time  < t_stop:
            poisson_train.append(last_time)
            next_isi = nextTime(rate)*1000.
            last_time += next_isi
    return poisson_train

#---------- From NE15 ----------#



def split_pos_neg_spikes(img_w, img_h, spike_pattern):
    pos = spike_pattern[:img_w*img_h]
    neg = spike_pattern[img_w*img_h:]
    return pos, neg



def out_to_spike_array(spikes, num_neurons):
    spike_array = [ [] for i in range(num_neurons) ] 
    nidx, ntime = 0, 1
    for spk in spikes:
        spike_array[ int(spk[nidx]) ].append( spk[ntime] )

    return spike_array

def get_max_time(pattern):
    max_t = float64(0)
    for times in pattern:
        for t in times:
            if t > max_t:
                max_t = t
    
    return float64(max_t)


def move_in_time(time_diff, pattern):
    time_diff = float64(time_diff)
    tmp = []
    i = 0
    for times in pattern:
        tmp.append([])
        
        for t in times:
            t = float64(t)
            t += time_diff
            tmp[i].append(t)
            
        i += 1
    # print("in move in time    ", len(tmp), len(pattern))
    return tmp




def merge_top_bottom(top_spks, bottom_spks):
    pad = len(top_spks)
    total_n = pad + len(bottom_spks)
    out_spks = [[] for i in range(total_n)]
    n_idx = 0
    for spk_times in top_spks:
        for t in spk_times:
            t = float64(t)
            out_spks[n_idx].append(t)
        n_idx += 1
        
    n_idx = pad
    for spk_times in bottom_spks:
        for t in spk_times:
            t = float64(t)
            out_spks[n_idx].append(t)
        n_idx += 1

    return out_spks



def merge_patterns(spks0, spks1):
    if len(spks0) != len(spks1):
        raise Exception("In concat_spikes: spike arrays not equal length")
    num_neurons = len(spks0)
    tmp = [[] for i in range(num_neurons)]
    for i in range(num_neurons):
        tmp[i][:] = spks0[i] + spks1[i]

    return tmp



def repeat_pattern(spike_array, num_repeats, sep_time, pattern_length=None):
    sep_time = float64(sep_time)
    n_idx = 0
    num_neurons = len(spike_array)
    spk_times = []
    new_spikes = [[] for i in range(num_neurons)]
    base_t = float64(0)
    if pattern_length is None:
        max_t = float64(get_max_time(spike_array) + sep_time)
    else:
        max_t = float64(pattern_length + sep_time)
        
    for i in range(num_repeats):
        base_t = i*max_t
        for n_idx in range(num_neurons):
            spk_times[:] = spike_array[n_idx]

            for t in spk_times:
                t = float64(t)
                new_spikes[n_idx].append( t + base_t )
        
            new_spikes[n_idx].sort()

    return new_spikes



def all_but_one(num_neurons, skip, start_t, end_t, t_step=1):
    spikes = [[] for i in range(num_neurons)]
    start_t = int(start_t); end_t = int(end_t); t_step=int(t_step)
    # print(start_t, end_t, t_step)
    for t in range(start_t, end_t, t_step):
        t = float64(t)
        for i in range(num_neurons):
            if i == skip:
                continue
            spikes[i].append(t)
    
    return spikes



def horizontal(num_neurons, time_step, neuron_id=None, start_time=0, total_time=None):
    neuron_id = num_neurons//2 if neuron_id == None else neuron_id
    total_time = (num_neurons-1)*time_step if total_time == None else total_time
    
    spks = [[] for i in range(num_neurons)]
    
    for t in range(start_time, total_time, time_step):
        t = float64(t)
        spks[neuron_id].append(t)

    return spks



def vertical(num_neurons, spk_time):
    spk_time = float64(spk_time)
    return [[spk_time] for i in range(num_neurons)]



def diagonal(num_neurons, time_step, 
             directions={HORZ: LEFT_TO_RIGHT, VERT: TOP_TO_BOTTOM}):
    time_step = float64(time_step)
    # print(directions[HORZ], directions[VERT])
    spike_array = [[] for i in range(num_neurons)]

    t = 0 if directions[HORZ] == LEFT_TO_RIGHT else time_step*(num_neurons - 1)
    t = float64(t)
    dt = time_step if directions[HORZ] == LEFT_TO_RIGHT else -time_step
    dt = float64(dt)
    
    start_idx = 0 if directions[VERT] == BOTTOM_TO_TOP else num_neurons - 1
    end_idx = num_neurons if directions[VERT] == BOTTOM_TO_TOP else -1
    didx = 1 if directions[VERT] == BOTTOM_TO_TOP else -1
    
    for idx in range(start_idx, end_idx, didx):
        # print("n %d, t %d"%(idx, t))
        t = float64(np.round(t))
        spike_array[idx].append(t)
        t += dt
    
    return spike_array



def brackets(num_neurons, time_step, direction=GREATER_THAN):
    time_step = float64(time_step)
    top_n = num_neurons//2
    btm_n = num_neurons - top_n
    if direction == LESS_THAN:
        top_spks = diagonal(top_n, time_step, 
                            directions={HORZ: LEFT_TO_RIGHT, 
                                        VERT: TOP_TO_BOTTOM})
        btm_spks = diagonal(btm_n, time_step, 
                            directions={HORZ: LEFT_TO_RIGHT, 
                                        VERT: BOTTOM_TO_TOP})
    else:
        top_spks = diagonal(top_n, time_step, 
                            directions={HORZ: RIGHT_TO_LEFT, 
                                        VERT: TOP_TO_BOTTOM})
        btm_spks = diagonal(btm_n, time_step, 
                            directions={HORZ: RIGHT_TO_LEFT, 
                                        VERT: BOTTOM_TO_TOP})

    return merge_top_bottom(top_spks, btm_spks)



def times_symbol(num_neurons, time_step):
    time_step = float64(time_step)
    spks0 = diagonal(num_neurons, time_step, 
                     directions={HORZ: LEFT_TO_RIGHT, 
                                 VERT: TOP_TO_BOTTOM})
    spks1 = diagonal(num_neurons, time_step, 
                     directions={HORZ: LEFT_TO_RIGHT, 
                                 VERT: BOTTOM_TO_TOP})

    return merge_patterns(spks0, spks1)



def plus_symbol(num_neurons, time_step):
    time_step = float64(time_step)
    hrz = horizontal(num_neurons, time_step)
    vrt = vertical(num_neurons, num_neurons//2)

    return merge_patterns(hrz, vrt)



def line(num_neurons, start_time, end_time, start_neuron, end_neuron, time_step=1):
    time_step = float64(time_step)
    spike_array = [[] for i in range(num_neurons)]

    dn = end_neuron - start_neuron
    dt = float64(end_time - start_time)
    if dn == 0:
        return horizontal(num_neurons, time_step, start_time, start_time+end_time)
    if dt == 0:
        return vertical(num_neurons, start_time)
    
    m_inv  = float(dt)/float(dn)
    t_pad = start_time if dn > 0 else end_time
    t_pad = float64(t_pad)
    # print("m_inv %03.4f\tt_pad %03.4f\t%03.4f -> %03.4f"%(m_inv, t_pad, 
                                                          # t_pad + start_neuron*m_inv, 
                                                          # t_pad + end_neuron*m_inv))
    n_step = 1 if dn > 0 else -1
    for n in range(start_neuron, end_neuron + n_step, n_step):
        t = np.round( (t_pad + n*m_inv)*time_step )
        t = float64(t)
        spike_array[n].append(t)
        # print("\t\t%03.4f -> %03.4f"%(n, np.round(t_pad + n*m_inv)))
    
    return spike_array



def random_pattern(num_neurons, start_time, end_time):
    r_seed = np.uint32(time.time()*100000)
    np.random.seed( r_seed )
    spikes = [[] for i in range(num_neurons)]
    time_range = end_time - start_time
    times = np.random.random(size=num_neurons)*time_range
    times += start_time
    times = float64( np.round(times) )
    for n in range(num_neurons):
        spikes[n].append(times[n])

    return spikes



def jitter_pattern(spike_array, max_jitter=1):
    r_seed = np.uint32(time.time()*100000)
    np.random.seed( r_seed )
    len_spks = len(spike_array)
    
    to_jitter = np.random.random(size=len_spks) 
    jitter = (to_jitter <= 0.5)
    jitter -= (to_jitter <= 0.25)*2
    jitter *= max_jitter
    jitter = float64(jitter)
    t_idx = 0
    spikes = [[] for i in range(len_spks)]
    for n_idx in range(len_spks):
        t_idx = 0
        len_t = len(spike_array[n_idx])
        spikes[n_idx][:] = spike_array[n_idx]
        if len_t == 1:
            spikes[n_idx][t_idx] += jitter[n_idx]
        elif len_t > 1:
            t_idx = np.random.randint(0, len_t)
            spikes[n_idx][t_idx] += jitter[n_idx]
    
    return spike_array



def oclude_pattern(spike_array, oclude_prob=0.2):
    r_seed = np.uint32(time.time()*100000)
    np.random.seed( r_seed )
    len_spks = len(spike_array)
    to_oclude = np.random.random(size=len_spks) < oclude_prob
    spikes = [[] for i in range(len_spks)]
    for n_idx in range(len_spks):
        t_idx = 0
        len_t = len(spike_array[n_idx])
        spikes[n_idx][:] = spike_array[n_idx]
        if to_oclude[n_idx] == 1:
            if len_t == 1:
                    spikes[n_idx][:] = []
            elif len_t > 1:
                t_idx = np.random.randint(0, len_t)
                t = spikes[n_idx][t_idx]
                spikes[n_idx].remove(t)
    
    return spike_array




def label_spikes_from_to(labels, num_classes, 
                         start, end,
                         on_time_ms, off_time_ms, 
                         start_time,
                         num_spikes_per_class=1, 
                         inter_spike_interval=1,
                         reverse=False,
                         inhibitory=False,
                         poisson=False):

    spks = [[] for i in range(num_classes)]
    t = float(start_time)
    print("%d -> %d = %d"%(start, end, len(labels[start:end])))
    max_spike_time = on_time_ms+np.round(off_time_ms/2.)
    dt = 0
    
    start_class_idx = 0
    end_class_idx = num_classes
    lcount = 0
    if inhibitory:
        lcount_div = (num_classes-1)*num_spikes_per_class
    else:
        lcount_div = num_spikes_per_class

    for label in labels[start:end]:
        
        # print("label_spikes ", t)
        i = 0
        if not inhibitory:
            start_class_idx = label
            end_class_idx = label+1
        
        for class_idx in range(start_class_idx, end_class_idx):
            
            for i in range(num_spikes_per_class):
                # print(np.uint32(time.time()*1000000000))
                np.random.seed(np.uint32(time.time()*(10**10)))
                if class_idx != label or (not inhibitory):
                    dt = i*inter_spike_interval
                    if reverse:
                        dt = on_time_ms - dt
                    rand_dt = np.random.randint(-3, 4) #[-2, -1, 0, 1, 2] or [..., 3)
                    if 0 <= dt < max_spike_time:
                        lcount += 1
                        if (t + dt + rand_dt) in spks[class_idx]:
                            continue
                        spks[class_idx].append( int(t + dt + rand_dt) )

        t += on_time_ms + off_time_ms

    for class_idx in range(start_class_idx, end_class_idx):
        spks[class_idx].sort()

    # print(lcount/lcount_div, t)
    # sys.exit(0)
    return spks


def img_spikes_from_to(path, num_neurons, 
                       start_file_idx, end_file_idx, 
                       on_time_ms, off_time_ms, 
                       start_time, delete_before=0, ext='txt'):
    start = start_file_idx
    end   = end_file_idx
    spikes = []

    spk_files = glob.glob(os.path.join(path, "*.%s"%(ext)))
    spk_files.sort()
    # print(path)
    # print(len(spk_files))
    f = None
    spks = [ [] for i in range(num_neurons) ]
    t = float(start_time)
    for fname in spk_files[start:end]:
        # print(fname)
        # spks[:] = [ [] for i in range(num_neurons) ]
        f = open(fname, 'r')
        for line in f:
            np.random.seed(np.uint32(time.time()*(10**10)))
            rand_dt = np.random.randint(-3, 4) #[-2, -1, 0, 1, 2] or [..., 3)

            vals = line.split(' ')
            nrn_id, spk_time = int(vals[0]), int( float(vals[1]) + t )
            # print("id = %s, t = %s"%(vals[0], vals[1]))
            rspk_time = spk_time + rand_dt
            if rspk_time in spks[nrn_id]:
                continue
                
            if rspk_time > (t + on_time_ms):
                continue
            if rspk_time < delete_before:
                continue

            rspk_time -= delete_before
            spks[nrn_id].append(rspk_time)
        f.close()
        # print(fname, t)
        # t += on_time_ms + off_time_ms
        t += off_time_ms
        # spikes.append(spks)
        
    for nrn_id in range(num_neurons):
        nspk = len(spks[nrn_id])

        if nspk == 0:
            continue

        spks[nrn_id][:] = list(set(spks[nrn_id])) # remove duplicates
        
        # random noise (spike loss)
        # np.random.seed(np.uint32(time.time()*(10**10)))
        # to_remove = np.random.choice(np.arange(nspk), size=int(nspk*0.1),
                                     # replace=False) 
        # for i in sorted(to_remove, reverse=True):
            # del spks[nrn_id][i]
            
        spks[nrn_id].sort()

    return spks
