import numpy as np
import pylab
from ..pattern.pattern_generator import out_to_spike_array
import matplotlib.animation as animation
import cv2

SRC, DST, W, DLY = 0, 1, 2, 3
X, Y, Z = 0, 1, 2

def my_imshow(ax, img, cmap="Greys_r", interpolation="none", 
              vmin=None, vmax=None, no_ticks=True, colorbar=False):
    if no_ticks:
        remove_ticks(ax)
    return pylab.imshow(img, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)


def remove_ticks(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def get_bounding_rect(positions):
    min_z = positions[:, Z].min()
    max_z = positions[:, Z].max()
    min_x = positions[:, X].min()
    max_x = positions[:, X].max()
    size_x = np.int32(np.ceil(max_x) - np.floor(min_x))
    size_z = np.int32(np.ceil(max_z) - np.floor(min_z))
    
    return [min_x, 0, min_z], \
           [max_x, 0, max_z], \
           [size_x, size_z]




def get_weights_and_positions(nrn_id, weights, positions, connections):
    src_rows = np.where( connections[:,DST] == nrn_id)[0]

    if src_rows.size == 0:
        return None

    src_ids = connections[src_rows, SRC].astype(np.int32)
    conn_weights = weights[src_rows]
    poss = positions[src_ids, :]
    min, max, shape = get_bounding_rect(poss)

    if shape[0] == 0:
        shape[0] = 1
    if shape[1] == 0:
        shape[1] = 1

    kernel = {'weights': conn_weights,
              'positions': poss,
              'min': min,
              'max': max, 
              'shape': shape}

    return kernel




def weights_to_img(nrn_id, weights, src_neuron_pos, connections):
    # conns = np.array(connections)

    img_info = get_weights_and_positions(nrn_id, weights, 
                                         src_neuron_pos,
                                         connections)
    if img_info is None:
        return None
    # print(img_info)
    img = np.zeros(img_info['shape'])
    
    
    xs = np.int32( img_info['positions'][:, X] - \
                   img_info['min'][X] )
                   
    # print(xs)
    zs = np.int32( img_info['positions'][:, Z] - \
                   img_info['min'][Z] )
    
    
    # print(zs)
    # print(img[xs, zs].shape)
    # print(img_info['weights'].shape)
    img[xs, zs] = img_info['weights']
    
    return img




def layer_to_imgs(nrn_ids, weights, src_neuron_pos, connections):
    cs = np.array(connections)
    ws = np.array(weights)
    imgs = []
    for nrn_id in nrn_ids:
        imgs.append(weights_to_img(nrn_id, ws, src_neuron_pos, cs))
        
    return imgs




def plot_imgs(imgs, figs_per_row, save=False, filename=None, min_v=0., max_v=2.):
    import pylab

    i = 0
    n_imgs = len(imgs)
    fig = pylab.figure()
    n_rows = n_imgs/figs_per_row + 1
    for img in imgs:
        i += 1
        if img is None:
            continue

        pylab.subplot(n_rows, figs_per_row, i)
        pylab.imshow(img, interpolation='none', cmap='Greys_r', \
                     vmin=min_v, vmax=max_v)
        pylab.axis('off')
    
    if save and filename is not None:
        pylab.savefig(filename, dpi=300)
    else:
        pylab.show()





def plot_layer(lyr, scale_factor=0.5, exc_color=(0., 0., 1.), 
               inh_color=(1., 0., 0.)):
    from mayavi import mlab 
    if 'inh' in lyr.keys():
        mlab.points3d(lyr['inh'][:, 0], lyr['inh'][:, 2], lyr['inh'][:, 1], 
                      scale_factor=scale_factor, color=inh_color,
                      mode='sphere')

    if 'exc' in lyr.keys():
        mlab.points3d(lyr['exc'][:, 0], lyr['exc'][:, 2], lyr['exc'][:, 1], 
                      scale_factor=scale_factor, color=exc_color,
                      mode='sphere')



def plot_connections( conn_list, src_pop, dst_pop, color=(0., 0.4, 0.) ):
    from mayavi import mlab
    u = []; v = []; w = [];
    x = []; y = []; z = [];
    for conn in conn_list:
        src, dst, weight = conn[0], conn[1], conn[2]
        x[:] = []; y[:] = []; z[:] = []
        x.append(src_pop[src, 0])
        x.append(dst_pop[dst, 0])
        y.append(src_pop[src, 2])
        y.append(dst_pop[dst, 2])
        z.append(src_pop[src, 1])
        z.append(dst_pop[dst, 1])
        mlab.plot3d(x, y, z,
                    line_width=1., color=color, )
        #~ x.append(src_pop[src, 0])
        #~ y.append(src_pop[src, 2])
        #~ z.append(src_pop[src, 1])
        #~ 
        #~ u.append( dst_pop[dst, 0] )
        #~ v.append( dst_pop[dst, 2] )
        #~ w.append( dst_pop[dst, 1] )
#~ 
    #~ mlab.quiver3d(x, y, z, u, v, w, 
                  #~ line_width=1., color=color, 
                  #~ # representation="wireframe"
                  #~ mode='arrow',
                  #~ #scale_factor=0.01,
                  #~ 
                  #~ )
    


def sum_horz_kern(img_width, img_height, kernel_width, 
                   horz_weights, save_file=None):
    
    min_w = (img_width - (kernel_width - 1))
    min_h = (img_height - (kernel_width - 1))
    new_img = np.zeros((img_height, min_w))
    k_sum = 0
    k_idx = 0
    for r in range(img_height):

        for c in range(min_w):
            k_sum = 0
            for k in range(kernel_width):
                k_idx = r*min_w + c + k
                k_sum += horz_weights[k_idx]
                
            new_img[r, c] = k_sum
    

    fig = pylab.figure(figsize=(5, 5))
    
    ax = pylab.subplot(1, 1, 1)
    # ax.set_title("k%s"%(k_idx))
    my_imshow(ax, new_img)

    return new_img



def sum_sep_kern(img_width, img_height, kernel_width, kernel_images):
    min_w = (img_width - (kernel_width - 1))
    min_h = (img_height - (kernel_width - 1))
    new_img = np.zeros((min_h, min_w))

    for r in range(min_h):

        for c in range(min_w):
            k_idx = r*min_w + c
            new_img[r, c] = kernel_images[k_idx].sum()
    
    pylab.tick_params(axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off')
    pylab.tick_params(axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off')
    fig = pylab.figure(figsize=(5, 5))
    
    ax = pylab.subplot(1, 1, 1)
    # ax.set_title("k%s"%(k_idx))
    my_imshow(ax, new_img)

    
    return new_img



def parse_conns_to_dict(conns):
    conn_dict = {}
    for conn in conns:
        src, tgt, w, d = conn
        if not (tgt in conn_dict.keys()):
            conn_dict[tgt] = {}
        
        conn_dict[tgt][src] = w
        
    return conn_dict




def plot_sep_kern(img_width, img_height, kernel_width, 
                  horz_weights, vert_weights, 
                  num_cols = 4, save_file=None, pylab_w = 2.,
                  h_col_step=2, h_row_step=2):
    kernel_width = int(kernel_width)
    min_w = (img_width  - (kernel_width - 1))
    min_h = (img_height - (kernel_width - 1))
    
    num_vert = len(vert_weights)//kernel_width
    num_rows = num_vert//num_cols + 1
    
    hrz_dict = parse_conns_to_dict(horz_weights)
    
    imgs = [np.zeros((kernel_width, kernel_width)) for i in range(num_vert)]
    
    v_idx = 0; h_idx = 0
    w_v = 0.;  w_h = 0.
    r = 0;     c = 0
    kv = 0;    kh = 0
    subplot_idx = 0
    k_idx = 0

    prev_v_idx = 0
    for v_conn in vert_weights:
        v_idx = int(v_conn[1])
        h_idx = int(v_conn[0])
        w_v   = float(v_conn[2])
        
        kh = 0
        for src in sorted( hrz_dict[h_idx].keys() ):
            w_h = hrz_dict[h_idx][src]
            imgs[v_idx][kv, kh] = w_v*w_h

            kh += 1
            
        kv += 1
        if kv == kernel_width:
            kv = 0


    subplot_idx = 0
    fig = pylab.figure(figsize=(pylab_w*num_cols, pylab_w*num_rows))
    for k_idx in range(len(imgs)):
        subplot_idx += 1
        ax = pylab.subplot(num_rows, num_cols, subplot_idx)
        my_imshow(ax, imgs[k_idx])
            
        prev_v_idx = v_idx
        
    return imgs

    

def plot_1D_conv_conns(conv_conns, img_w, img_h, krnl_w, horiz_conv=True,
                       col_step=2, row_step=2, prev_col_step=2, prev_row_step=2):
    src = -1
    conn_str = ""
    conn_dic = {}
    for c in conv_conns:
        if not ( c[1] in conn_dic ):
            conn_dic[c[1]] = []
        conn_dic[c[1]].append((c[0], c[2]))


    max_h = img_h - krnl_w + 1
    max_w = img_w - krnl_w + 1
    if horiz_conv:
        num_kernels = (max_w//col_step)*(img_h//row_step)
        out_w = img_w
        out_h = img_h
    else:
        num_kernels = (max_w//col_step)*(max_h//row_step)
        out_w = max_w//prev_col_step
        out_h = img_h//prev_row_step

    
    imgs = [np.zeros(out_h*out_w) for i in range(num_kernels)]
    print(num_kernels)
    n_cols = 6
    n_rows = num_kernels//n_cols + 1
    fig = pylab.figure(figsize=(2.8*n_cols, 2.8*n_rows))
    subplot_idx = 1
    sources = []
    i = 0
    w = 1.
    for tgt in conn_dic.keys():
        ax = pylab.subplot(n_rows, n_cols, subplot_idx)
        sources[:] = [ i for i, w in conn_dic[tgt] ]
        ax.set_title("%s -> %s"%(sources, tgt))
        subplot_idx += 1
        
        imgs[tgt][:] = 0
        for i, w in conn_dic[tgt]:
            imgs[tgt][i] = w
        
        my_imshow(ax, imgs[tgt].reshape((out_h, out_w)))



def plot_spikes(spike_array, max_y=None, pad = 2, title="", marker='.', 
                color='blue', markersize=4, base_id=0, plotter=pylab):
    neurons = []
    times   = []
    n_idx   = 0
    min_time = 9999
    max_time = -9999
    for spike_times in spike_array:
        for t in spike_times:
            if t < min_time:
                min_time = t
            if t > max_time:
                max_time = t
            neurons.append(n_idx + base_id)
            times.append(t)
        n_idx += 1
    
    n_idx += base_id
    plotter.plot(times, neurons, linestyle='none',
                 marker=marker, markerfacecolor=color, markersize=markersize,
                 color=color)
    # pylab.xlim(min_time - pad, max_time + pad)
    # print("max_y ", max_y)
    if max_y is None:
        
        max_y = n_idx 
    
    if plotter != pylab:
        plotter.set_ylim(-pad, max_y + pad)
    else:
        plotter.ylim(-pad, max_y + pad)


    # pylab.xlabel("Time (ms)")
    # pylab.ylabel("Neuron id")
    # pylab.title(title)




def plot_output_spikes(spikes, pad=0, marker='.', color='blue', markersize=4,
                       plotter=pylab, max_y=None, pad_y=2, markeredgewidth=0,
                       markeredgecolor='none'):
    if len(spikes) == 0:
        return 0
        
    spike_times = [spike_time for (neuron_id, spike_time) in spikes]
    spike_ids  = [neuron_id for (neuron_id, spike_time) in spikes]
    # print(np.max(spike_times), np.min(spike_times))
    # print(np.max(spike_ids), np.min(spike_ids))
    max_id = 0 if len(spike_ids) == 0 else np.max(spike_ids)

    if max_y is None:
        max_y = max_id

    spike_ids[:] = [neuron_id + pad for neuron_id in spike_ids]

    plotter.plot(spike_times, spike_ids, marker, markersize=markersize, 
                 markerfacecolor=color, markeredgecolor=markeredgecolor, 
                 markeredgewidth=markeredgewidth)
    if plotter != pylab:
        plotter.set_ylim(-pad_y, max_y + pad_y)
    else:
        plotter.ylim(-pad_y, max_y + pad_y)
        
    return pad + max_id + 1



def spikes_in_time_range(spikes, from_t, to_t, start_idx=0):
    spks = []
    end_idx = start_idx
    for spk in spikes[start_idx:]:
        if from_t <= spk[1] <= to_t:
            spks.append(spk)
        
        end_idx += 1
        
        if spk[1] > to_t:
            break
        
    return spks, end_idx



def img_from_spikes(spikes, img_width, height):
    img = np.zeros((img_height*img_width), dtype=np.uint8)
    for spk in spikes:
        img[spk[0]] += 1
    
    return img.reshape((img_height, img_width))

def default_img_map(i, w, h):
    row = i//w
    col = i%w
    
    return row, col, 1

def imgs_in_T_from_spike_array(spike_array, img_width, img_height, 
                               from_t, to_t, t_step, out_array=False,
                               thresh=12, up_down = None,
                               map_func = default_img_map):
    num_neurons = img_width*img_height
    if out_array: # should be output spike format
        mult = 2 if up_down is None else 1
        print("imgs_in_T, num neurons: %d"%(mult*num_neurons))
        spike_array = out_to_spike_array(spike_array, mult*num_neurons)


    
    spikes = [ sorted(spk_ts) for spk_ts in spike_array ]
        
    imgs = [ np.zeros((img_height, img_width, 3), dtype=np.uint8) \
             for t in range(from_t, to_t, t_step) ]
    
    nrn_start_idx = [ 0 for t in range(len(spikes)) ]
    
    t_idx = 0
    for t in range(from_t, to_t, t_step):
        for nrn_id in range(len(spikes)):
            
            if spikes[nrn_id] is None:
                continue
            
            if len(spikes[nrn_id]) == 0:
                continue
                
            nrn_id = np.uint32(nrn_id)
            row, col, up_dn = map_func(nrn_id, img_width, img_height)
            if up_down is not None:
                up_dn = up_down
                
            for spk_t in spikes[nrn_id][nrn_start_idx[nrn_id]:]:
                if t <= spk_t < t+t_step:
                    # imgs[t_idx][row, col, up_dn] += thresh
                    imgs[t_idx][row, col, up_dn] = 255
                else:
                    break
                nrn_start_idx[nrn_id] += 1

        t_idx += 1

    # for t_idx in range(len(imgs)):
        # imgs[t_idx] = imgs[t_idx].reshape((img_height, img_width))
    
    return imgs



def img_from_spike_array(spike_array, img_width, img_height):
    img = np.zeros((img_height*img_width))
    nrn_idx = 0
    for spike_times in spike_array:
        for t in spike_times:
            img[nrn_idx] += 1
        nrn_idx += 1

    return img.reshape((img_height, img_width))




def plot_class_weights_2D_array(weights, num_classes, img_width, img_height, 
                               num_cols, num_rows, vmin=None, vmax=None, 
                               hex_plot=False):
    
    num_prev_neurons = img_width*img_height
    
    for i in range(num_classes):
        ax = pylab.subplot(num_rows, num_cols, i+1)
        if hex_plot:
            pass
            # from ....hex_net.hex_pixel import HexImage
            # from ....hex_net.vis.hex_plot import remove_ticks, plot_hex_img
            # plot_hex_img(ax, HexImage( weights[:, i].reshape((img_height, img_width)) ) \
            #              vmin=vvmin, vmax=vmax, markersize=5)
        else:
            my_imshow(ax, weights[:, i].reshape((img_height, img_width)),
                      vmin=vmin, vmax=vmax)


def plot_class_weights(weights, num_classes, img_width, img_height, 
                       num_cols, num_rows, vmin=None, vmax=None):
    num_prev_neurons = img_width*img_height
    imgs = [np.zeros((num_prev_neurons)) for i in range(num_classes)]
    

    for j in range(num_classes):
        for i in range(num_prev_neurons):
            # print(i*num_classes + j, i, j)
            imgs[j][i] = weights[i*num_classes + j]

    sub_idx = 1
    for img in imgs:
        ax = pylab.subplot(num_rows, num_cols, sub_idx)
        sub_idx += 1
        my_imshow(ax, img.reshape((img_height, img_width)),
                  vmin=vmin, vmax=vmax)
    
    return imgs


def remove_ticks(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])




def build_gif(imgs, filename='', show_gif=True, save_gif=True, title='',
              interval=100):

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    ims = map(lambda x: (ax.imshow(x, interpolation='none'), \
                         ax.set_title(title)), imgs)

    im_ani = animation.ArtistAnimation(fig, ims, interval=interval, \
                                       repeat_delay=0, blit=False)

    if save_gif:
        if filename == '':
            filename = 'animation.gif'
        im_ani.save(filename, writer='imagemagick')

    if show_gif:
        pylab.draw()

    return





# === ------------------------------------------------------------ === #



