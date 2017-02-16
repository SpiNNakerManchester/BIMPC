from ..common import *
from mapping_funcs import row_col_to_input

def full_kernel_connector(pre_layer_width, pre_layer_height, kernel, 
                          exc_delay=2., inh_delay=1.,
                          col_step=1, row_step=1, 
                          col_start=0, row_start=0, 
                          map_to_src=row_col_to_input,
                          row_bits=8, pop_width=None,
                          on_path=True,
                          min_w = 0.001, remove_inh_only=True):
    '''Create connection list based on a convolution kernel, the format
       for the lists is to be used with PyNN 0.7. 
       (Pre neuron index, Post neuron index, weight, delay)
       
       :param layer_width: Pre layer width
       :param layer_height: Pre layer height
       :param kernel: Convolution kernel
       :param col_step: Skip this many columns (will reduce Post layer size)
       :param row_step: Skip this many rows (will reduce Post layer size)
       :param delay: How many time units will it take the spikes to
                     get from Pre to Post

       :return exc_conns: Excitatory connections list
       :return inh_conns: Inhibitory connections list
    '''
    row_bits = row_bits if pop_width is None else pop_width
    
    layer_width, layer_height = pre_layer_width, pre_layer_height
    exc_conns = []
    inh_conns = []
    kh, kw = kernel.shape
    half_kh, half_kw = kh//2, kw//2
    
    dst_width  = subsamp_size(col_start, layer_width, col_step)
    dst_height = subsamp_size(row_start, layer_height, row_step)
    num_dst = dst_width*dst_height
    exc_counts = [0 for dr in range(num_dst)]
    inh_counts = [0 for dr in range(num_dst)]
    
    for dr in range(row_start, layer_height, row_step):
        for dc in range(col_start, layer_width, col_step):

            sr0 = dr - half_kh
            sc0 = dc - half_kw
            drr = subsamp_size(row_start, dr, row_step)
            dcc = subsamp_size(col_start, dc, col_step)
            dst = drr*dst_width + dcc

            for kr in range(kh):
                sr = sr0 + kr
                if sr < 0 or sr >= layer_height:
                    continue

                for kc in range(kw):
                    sc = sc0 + kc
                    if sc < 0 or sc >= layer_width:
                        continue

                    w = kernel[kr, kc]
                    if np.abs(w) < min_w:
                        continue
                    
                    src = map_to_src(sr, sc, on_path, row_bits)

                    # src = sr*layer_width + sc + src_start_idx
                    # divide values so that indices match the size of the
                    # Post (destination) next layer
                    
                    if dst >= num_dst:
                        continue
                        
                    src = np.uint32(src) 
                    dst = np.uint32(dst)
                    w = float(w);

                    if w < 0:
                        inh_conns.append((src, dst, w, inh_delay))
                        inh_counts[dst] += 1
                    elif w > 0:
                        exc_conns.append((src, dst, w, exc_delay))
                        exc_counts[dst] += 1
    
    if remove_inh_only:
        exc_conns[:], inh_conns[:] = remove_inh_only_dst(exc_conns, inh_conns, 
                                                         exc_counts)
                                                         
    
    return exc_conns, inh_conns



def remove_inh_only_dst(exc_conns, inh_conns, exc_counts):
    new_exc = exc_conns[:] #copy lists --- paranoid choice
    new_inh = inh_conns[:] #copy lists --- paranoid choice

    for i in range(len(exc_counts)):
        if exc_counts[i] == 0:
            new_exc[:] = [x for x in new_exc if x[1] != i]
            new_inh[:] = [x for x in new_inh if x[1] != i]

    return new_exc, new_inh



def inh_neighbours(r, c, row_step, col_step, kw, kh, correlation,
                   delay=1, selfdelay=4):
    if row_step >= kh or col_step >= kw:
        return []
    else:
        hlf_kw = kw//2
        hlf_kh = kh//2
        src = (r//row_step)*(imgw//col_step) + c//col_step
        
        nbr = []
        kr = 0
        for nr in range(r - hlf_kh, r + hlf_kh + 1, row_step):
            if nr < 0 or nr >= imgh:
                kr += 1
                continue
                
            kc = 0
            for nc in range(c - hlf_kw, c + hlf_kw + 1, col_step):
                if nc < 0 or nc >= imgw:
                    kc += 1
                    continue
                
                if nr == r and nc == c:
                    d = selfdelay
                    dst = src
                else:
                    d = delay
                    dst = (nr//row_step)*(imgw//col_step) + nc//col_step

                w = correlation[kr, kc]
                
                nbr.append( (src, dst, d, w) )
                
                kc += 1
            kr += 1
        
        return nbr


def lateral_inh_connector(layer_width, layer_height, correlation,
                          inh_weight, delay=1., self_delay=4,
                          col_step=1, row_step=1, 
                          col_start=0, row_start=0, min_w = 0.001):
    '''Assuming there's an interneuron layer with as many neurons as dest'''
    # exc_conns = []
    inh_conns = []
    kh, kw = correlation.shape
    half_kh, half_kw = kh//2, kw//2
    num_dst = ( (layer_height - row_start)//row_step )*\
              ( (layer_width  - col_start)//col_step )
              
    for dr in range(row_start, layer_height, row_step):
        for dc in range(col_start, layer_width, col_step):
            sr0 = dr - half_kh
            sc0 = dc - half_kw

            for kr in range(kh):
                sr = sr0 + kr
                if sr < 0 or sr >= layer_height:
                    continue

                for kc in range(kw):
                    sc = sc0 + kc
                    if sc < 0 or sc >= layer_width:
                        continue

                    #divide values so that indices match the size of the
                    #Post (destination) next layer
                    ### from Post to Interneurons
                    # src = (dr//row_step)*layer_width//col_step + (dc//col_step)
                    # exc_conns.append( (src, src, exc_weight, delay) )
                    
                    ### from Interneurons to Post
                    inh_conns += inh_neighbours(dr, dc, row_step, col_step, kw, kh,
                                                correlation, delay, selfdelay)
                    
    # return exc_conns, inh_conns
    return inh_conns
                        
                    

                    
