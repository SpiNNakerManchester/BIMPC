from ..common import *
from standard_connectors import *

CONN_TYPES = ['all2all', 'prob', 'all_inputs']
PIX, ORIENT, DIR = 0, 1, 2

def prob_connector(pre, post, conn_prob, weight=1., delay=1.):

    np.random.seed(np.uint32(time.time()*1000000))
    dice = np.random.random( size=(len(pre), len(post)) )
    
    np.random.seed(np.uint32(time.time()*1000000))
    ws = np.random.normal( loc=weight, scale=0.5*weight, \
                           size=(len(pre), len(post)) )
    # ws = np.abs(ws)
    rows, cols = np.where(dice <= conn_prob)
    
    conns = [ (pre[rows[i]], post[cols[i]], np.abs(ws[rows[i], cols[i]]), delay) \
              for i in range(len(rows)) ]
    
    return conns


def convolution_wta(layer_width, layer_height, convolution_width,
                    neurons_per_zone, input_weight, 
                    wta_exc_weight, wta_inh_weight, 
                    all2all=False, conn_prob=0.1,
                    col_start=0, row_start=0,
                    col_step=1, row_step=1):

    convw = convolution_width
    lh, lw = layer_height, layer_width
    zone_count = 0
    nrn_pz = neurons_per_zone
    
    exc_conns = []
    wta_e_conns = []
    wta_i_conns = []
    
    for r in range(row_start, lh-convw+1, row_step):
        for c in range(col_start, lw-convw+1, col_step):
            src = [i*lw + j for i in range(r, r + convw) \
                            for j in range(c, c + convw)]
            # print(src)
            dst = [zone_count*nrn_pz + i for i in range(neurons_per_zone)]

            zone_count += 1 
            
            if all2all:
                exc_conns += list_all2all(src, dst, input_weight, delay=1)
            else:
                exc_conns += prob_connector(src, dst, conn_prob, \
                                            input_weight, delay=1)
            
            # assume there's a population equal to dst for interneurons
            tmp = list_wta_interneuron(dst, dst, wta_exc_weight, 
                                       wta_inh_weight, delay=1.)
            
            wta_e_conns += tmp[0]
            wta_i_conns += tmp[1]

    return exc_conns, wta_e_conns, wta_i_conns


def influence_conv_wta(widths, heights, convolution_width,
                       neurons_per_zone, input_weight, 
                       wta_exc_weight, wta_inh_weight, 
                       all2all=False, conn_prob=0.1,
                       col_start=0, row_start=0,
                       col_step=1, row_step=1):

    convw = convolution_width
    lh, lw = heights[0], widths[0]
    zone_count = 0
    nrn_pz = neurons_per_zone
    
    exc_conns = []
    inf_conns = [ [] for i in range(len(widths) - 1) ]
    wta_e_conns = []
    wta_i_conns = []
    
    for r in range(row_start, lh-convw, row_step):
        for c in range(col_start, lw-convw, col_step):
            
            src = [i*lw + j for i in range(r, convw) \
                            for j in range(c, convw)]
            dst = [zone_count*nrn_pz + i for i in range(neurons_per_zone)]

            zone_count += 1 
            
            if all2all:
                exc_conns += list_all2all(src, dst, input_weight, delay=1)
            else:
                exc_conns += prob_connector(src, dst, input_weight, delay=1)
            
            # assume there's a population equal to dst for interneurons
            tmp = list_wta_interneuron(dst, dst, wta_exc_weight, 
                                       wta_inh_weight, delay=1.)
            
            wta_e_conns += tmp[0]
            wta_i_conns += tmp[1]

    return exc_conns, wta_e_conns, wta_i_conns
