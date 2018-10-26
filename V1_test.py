
# coding: utf-8

# In[1]:

# get_ipython().magic(u'matplotlib inline')
# get_ipython().magic(u'reload_ext autoreload')
# get_ipython().magic(u'autoreload 2')

import numpy as np
import pylab as plt
# import vision.sim_tools.connectors.kernel_connectors as kconn
from vision.spike_tools.vis import my_imshow, plot_spikes,                                    plot_output_spikes,                                    imgs_in_T_from_spike_array,                                    build_gif
# from vision.sim_tools.vis import plot_connector_3d
# import vision.sim_tools.kernels.center_surround as csgen
# import vision.sim_tools.kernels.gabor as gabgen
import vision.sim_tools.connectors.mapping_funcs as mapfun

from mpl_toolkits.mplot3d import axes3d, Axes3D

from vision.retina import Retina, dvs_modes, MERGED
from vision.lgn import LGN
from vision.v1 import V1
from vision.spike_tools.pattern import pattern_generator as pat_gen
import pickle
import cv2
import os
import sys

# from pyNN import nest as sim
from pyNN import spiNNaker as sim

print(sim.__name__)

on_imgs = []
off_imgs = []

# In[2]:
def get_spikes(pop, lbl=''):
    spks = []
    try:
        spks[:] = pop.getSpikes(compatible_output=True)
    except:
        print("no spikes for population: %s"%(lbl))
        
    return spks

def output_to_spike_source_array(spikes, num_neurons):
    ssa = [[] for i in range(num_neurons)]
    max_i = -1
    for (nrn_id, spk_t) in spikes:
        if max_i < nrn_id:
            max_i = int(nrn_id)
            
        ssa[int(nrn_id)].append(spk_t)
    # print(max_i)
    
    for i in range(num_neurons):
        ssa[i][:] = sorted(ssa[i])
    
    return ssa
    
    
def setup_cam_pop(sim, spike_array, img_w, img_h, w2s=1.6):
    print("Setting up camera population...")
    pop_size = img_w*img_h*2
    cell = sim.IF_curr_exp
    params = { 'cm': 0.2,  # nF
               'i_offset': 0.0,
               'tau_m': 10.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 2.,
               'tau_syn_I': 2.,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
             }
    dmy_pops = []
    dmy_prjs = []
    
    if sim.__name__ == 'pyNN.spiNNaker':
        cam_pop = sim.Population(pop_size, sim.SpikeSourceArray,
                                 {'spike_times': spike_array},
                                 label='camera')
    else:
        cam_pop = sim.Population(pop_size, cell, params,
                                 label='camera')
        for i in range(pop_size):
            dmy_pops.append(sim.Population(1, sim.SpikeSourceArray, 
                                           {'spike_times': spike_array[i]},
                                           label='pixel (row, col) = (%d, %d)'%\
                                           (i//img_w, i%img_w)))
            conn = [(0, i, w2s, 1)]
            dmy_prjs.append(sim.Projection(dmy_pops[i], cam_pop,
                                           sim.FromListConnector(conn),
                                           target='excitatory'))
    
    
    return cam_pop, dmy_pops, dmy_prjs



def plot_out_spikes(on_spikes, off_spikes, img_w, img_h, 
                    end_t_ms, ftime_ms, thresh, title):
    if len(on_spikes) == 0:
        return 
    
    up = None if len(off_spikes) == 0 else 1

    on_imgs[:] = imgs_in_T_from_spike_array(on_spikes, img_w, img_h, 
                                         0, end_t_ms, ftime_ms, 
                                         out_array=True, thresh=thresh, 
                                         up_down=up)
    if len(off_spikes) > 0:
        off_imgs[:] = imgs_in_T_from_spike_array(off_spikes, img_w, img_h, 
                                              0, end_t_ms, ftime_ms, 
                                              out_array=True, thresh=thresh,
                                              up_down=0)
    else:
        off_imgs[:] = [i for i in on_imgs]
    fps = 50.
    mspf = int(1000./fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    title = title.replace(' ', '_')
    title = title.replace(':', '--_')
    scl = 20
    vid_shape = (img_w*scl, img_h*scl)
    # vid_shape = (img_h*scl, img_w*scl)
    vid_out = cv2.VideoWriter("%s.m4v"%title, fourcc, fps, vid_shape)
    num_imgs = len(on_imgs)
    # cols = 10
    # rows = num_imgs//cols + 1
    # figw = 1
    # fig = plt.figure(figsize=(figw*cols, figw*rows))
    for i in range(num_imgs):

        off_imgs[i][:,:,1] = on_imgs[i][:,:,1]
        vid_out.write( cv2.resize(cv2.cvtColor(off_imgs[i], cv2.COLOR_BGR2RGB), vid_shape, 
                                  interpolation=cv2.INTER_NEAREST) )
        # ax = plt.subplot(rows, cols, i+1)
        # my_imshow(ax, off_imgs[i], cmap=None)

    # plt.suptitle(title)
    # plt.draw()
    # plt.savefig("frames_for_%s.png"%title, dpi=150)
    # plt.close()
    
    # build_gif(off_imgs, "%s.gif"%title, show_gif=True, title=title, interval=mspf)
    # plt.close()
    
    vid_out.release()
    
    return


# In[3]:
do_lgn = True
do_v1  = False and do_lgn
learning_on = True
learning_off = not learning_on
# img_w, img_h = 160, 128
# img_w, img_h = 80, 64
img_w, img_h = 40, 32
step2  = 3
step4  = 6
num_neurons = img_w*img_h*2
fps = 50
frames = 60
thresh = 12
deg = 135
dx = 1.
on_time_ms  = 15000 #int( frames*(1000./fps) )
ftime_ms    = int( 1000./fps )
off_time_ms = 0
start_time  = 10


# In[4]:

spks = pickle.load(open('breakout_test_output_spikes.pickle', 'r'))
spk_sa = output_to_spike_source_array(spks, img_w*img_h*2)


# In[5]:


w2s =3.
sim.setup(timestep=1., max_delay=144., min_delay=1.)
if sim.__name__ == 'pyNN.spiNNaker':
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 70)
    


cam, dmy_ssa_cam, dmy_prj_cam = setup_cam_pop(sim, spk_sa, 
                                              img_w, img_h, w2s=w2s)
print(cam.size)
cam.record()

burst_params  = {'a': 0.02, 'b': 0.25, 'c': -55.0, 'd': 0.05, 
                 'v_init': -64., 'u_init': -64*0.25, 'tau_syn_E': 8.}
class2_params = {'a': 0.2, 'b': 0.26, 'c': -65.0, 'd': 0.0, 
                 'v_init': -64., 'u_init': -64*0.26}
relay = sim.Population(img_w*img_h*2, sim.IZK_curr_exp, 
                       burst_params, label='relay')
relay.record()
sim.Projection(cam, relay, sim.OneToOneConnector(weights=2.))

cfg = {'record': {'voltages': False, 
                  'spikes': True,
                 },
       'row_bits': 5,
      }
mode = dvs_modes[MERGED]
retina = Retina(sim, relay, img_w, img_h, mode, cfg=cfg)
retina.pops['on']['cam_inter'].record()
retina.pops['off']['cam_inter'].record()
print("RETINA SIZES --------------------------------------------")
for k in retina.shapes:
    print(k)
    print(retina.shapes[k])
print("RETINA CONNECTORS +++++++++++++++++++++++++++++++++++++++++++++")
for c in retina.conns:
    for k in retina.conns[c]:
        print(c, k)
        if 'dir' in k:
            for conn in retina.conns[c][k]:
                for ccc in conn:
                    print(ccc)
# sys.exit()

if do_lgn:
    cfg = {'record': {'voltages': False, 
                      'spikes': True,
                     },
      }
    lgn = LGN(sim, retina, cfg=cfg)
    
if do_v1:
    cfg = {'record': {'voltages': False, 
                  'spikes': True,
                 },
    }

    v1 = V1(sim, lgn, learning_off, cfg=cfg)

run_time = on_time_ms

print("Start run for %s ms"%run_time)
sim.run(run_time)

cam_spks = {}
cam_spks['on'] = get_spikes(retina.cam['on'], 'cam_on')
cam_spks['off'] = get_spikes(retina.cam['off'], 'cam_off')
inter_cam_spks = {}
inter_cam_spks['on'] = get_spikes(retina.pops['on']['cam_inter'], 
                                  'cam_inter_on')
inter_cam_spks['off'] = get_spikes(retina.pops['off']['cam_inter'], 
                                   'cam_inter_off')


print("Trying to get output spikes")
out_spks = {}
print("\tFor Retina")
for k in retina.pops.keys():
    out_spks[k] = {}
    for p in retina.pops[k].keys():
        out_spks[k][p] = {}
        if isinstance(retina.pops[k][p], dict):
            for t in retina.pops[k][p].keys():
                key = "retina_%s__%s__%s"%(k, p, t)
                out_spks[k][p][t] = get_spikes(retina.pops[k][p][t], key)
# 
if do_lgn:
    lgn_spks = {}
    print("\tFor LGN")
    
    for c in lgn.pops.keys():
        lgn_spks[c] = {}
        for k in lgn.pops[c].keys():
            key = "lgn_%s_%s"%(c,k)
            lgn_spks[c][k] = get_spikes(lgn.pops[c][k]['output'], key)
            

if do_v1:
    v1_spks = {}
    v1_inh = {}
    print("\tFor V1 simples")
    for r in v1.simple:
        v1_spks[r] = {}
        v1_inh[r] = {}
        for c in v1.simple[r]:
            key = "v1_spl_%s_%s"%(r, c)
            v1_spks[r][c] = get_spikes(v1.simple[r][c].pops['simple'], key)
            key = "v1_inh_%s_%s"%(r, c)
            v1_inh[r][c]  = get_spikes(v1.simple[r][c].pops['wta_inh'], key)

    pickle.dump(v1_spks, open('v1_spikes.pickle', 'w'))

sim.end()

print("-------------------------------------------------------------------")
print("\t\tDone simulation!!!")
print("-------------------------------------------------------------------")


# In[ ]:

# print(retina.conns['on']['cs4'][0])


# # plot RETINA

# In[ ]:


plt.figure()
plot_output_spikes(cam_spks['off'], color='red', markersize=3, 
                   marker='_', markeredgewidth=1, markeredgecolor='red')
plot_output_spikes(cam_spks['on'], color='green', markersize=3, 
                   marker='|', markeredgewidth=1, markeredgecolor='green')
plt.suptitle("CAMERA SPIKES")
plt.savefig("camera_spikes.png", dpi=150)
plt.draw()
plt.close()

#########################
# def mf(nid, width, height):
    # row, col, up_dn = 0, 0, 0
    # row   = (nid >> 1) & 63
    # col   = (nid >> 7) & 63
    # up_dn = nid & 1
    # 
    # #print(row, col, up_dn)
    # #col = width  - 1 if col >= width  else col
    # #row = height - 1 if row >= height else row
    # return row, col, up_dn
# 
# plt.figure()
# on_imgs[:] = imgs_in_T_from_spike_array(cam_spks['on'], img_w, img_h, 
                                        # 0, on_time_ms, ftime_ms, 
                                        # out_array=True, thresh=thresh, 
                                        # map_func=mf)
# num_imgs = len(on_imgs)
# cols = 10
# rows = num_imgs//cols + 1
# figw = 1
# fig = plt.figure(figsize=(figw*cols, figw*rows))
# for i in range(num_imgs):
    # ax = plt.subplot(rows, cols, i+1)
    # my_imshow(ax, on_imgs[i], cmap=None)
# plt.show()
#########################

plt.figure()
plot_output_spikes(inter_cam_spks['off'], color='red', markersize=3, 
                   marker='_', markeredgewidth=1, markeredgecolor='red')
plot_output_spikes(inter_cam_spks['on'], color='green', markersize=3, 
                   marker='|', markeredgewidth=1, markeredgecolor='green')
plt.suptitle("INTER CAMERA SPIKES")
plt.savefig("inter_camera_spikes.png", dpi=150)
plt.draw()
plt.close()

if 'on' in out_spks:
    for p in sorted(out_spks['on'].keys()):
        for t in out_spks['on'][p].keys():
            if not out_spks['on'][p][t] and \
               not out_spks['off'][p][t]:
                continue
               
            if t != 'ganglion':
                continue
            # if t != 'bipolar':
                # continue
                
    #         if p == 'cs' or p == 'cs2' or p == 'cs4':
    #             krn = retina.cs
    #             fig = plt.figure(figsize=(1,1))
    #             ax = plt.subplot(1,1,1)
    #             my_imshow(ax, krn)
    # #         else:
    # #             krn = retina.gab[p]
    #         print("%s, %s"%(p, t))
    #         print(len(out_spks['on'][p][t]))
    #         print(len(out_spks['off'][p][t]))
            
            if 'cs' in p:
                w = retina.shapes[p]['width']
                h = retina.shapes[p]['height']
            elif 'gabor' in p:
                w = retina.shapes['gabor']['width']
                h = retina.shapes['gabor']['height']
            elif 'dir' in p:
                w = retina.shapes['dir']['width']
                h = retina.shapes['dir']['height']
            else: 
                w = 0
                h = 0

            fig = plt.figure()#figsize=(16, 18))
            plot_output_spikes(out_spks['on'][p][t], marker='x', markeredgewidth=1., 
                               markeredgecolor='g', color='g')
            plot_output_spikes(out_spks['off'][p][t], marker='|', markeredgewidth=1., 
                               markeredgecolor='r', color='r')
            plt.suptitle("%s, %s"%(p, t))
            plt.draw()
            plt.savefig("retina_%s_%s_spikes.png"%(p, t), dpi=150)
            plt.close()
            
            plot_out_spikes(out_spks['on'][p][t], 
                            out_spks['off'][p][t], 
                            w, h, 
                            on_time_ms, ftime_ms, 
                            thresh=thresh, 
                            title="retina_%s_%s"%(p, t))
            plt.close()
        
        


# # plot LGN

# In[ ]:

if do_lgn and 'on' in lgn_spks:
    for k in sorted(lgn_spks['on'].keys()):
        if not lgn_spks['on'][k] and \
           not lgn_spks['off'][k]:
            continue
        
        if k == 'cs4':
            w = retina.filter_width4
            h = retina.filter_height4
        elif k == 'cs2':
            w = retina.filter_width2
            h = retina.filter_height2
        else:
            w = img_w
            h = img_h
        
        fig = plt.figure()#figsize=(16, 18))
        
        plot_output_spikes(lgn_spks['on'][k], color='g')
        plot_output_spikes(lgn_spks['off'][k], color='r')
        plt.suptitle("LGN %s"%(k))
        plt.draw()
        plt.savefig("lgn_%s_spikes.png"%(k), dpi=150)
        plt.close()
        
        # plot_out_spikes(lgn_spks['on'][k],
                        # lgn_spks['off'][k],
                        # w, h, 
                        # on_time_ms, ftime_ms, 
                        # thresh=thresh, 
                        # title="LGN_%s"%(k))
        # plt.close()

# In[ ]:

if do_v1:
    for r in v1_spks:
        for c in v1_spks[r]:
            
            # print(v1_spks[r][c])
            if not v1_spks[r][c]:
                continue
                
            fig = plt.figure()#figsize=(16, 18))
            plot_output_spikes(v1_spks[r][c], color='b')
            plt.suptitle("V1 simple (%d, %d)"%(r, c))
            plt.draw()
            plt.savefig("v1_simple_(%d_%d)_spikes.png"%(r, c), dpi=150)
            plt.close()


            fig = plt.figure()#figsize=(16, 18))
            plot_output_spikes(v1_inh[r][c], color='r')
            plt.suptitle("V1 inh (%d, %d)"%(r, c))
            plt.draw()
            plt.savefig("v1_inh_(%d_%d)_spikes.png"%(r, c), dpi=150)
            plt.close()

# In[ ]:



