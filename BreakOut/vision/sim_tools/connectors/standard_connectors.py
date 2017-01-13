from ..common import *
def default_mapping(r, c, ud, w, h):
    return r*w + c

def subsample(in_width, in_height, width_sub, height_sub, is_up, 
              weight=2., delay=1., coord_mapping=default_mapping):
    ''':param width_sub: pixel window width
       :param height_sub: pixel window height
    '''
    out_width  = in_width//width_sub
    out_height = in_height//height_sub
    conns = []
    for r in range(in_height):
        for c in range(in_width):
            src = coord_mapping(r, c, up_down, in_width, in_height)
            dst = (r//height_sub)*out_width + c//width_sub,

            conns.append(( src, dst, weight, delay ))
    
    return conns


def all2all(num_pre, num_post, weight=2., delay=1., start_idx_pre=0, 
            start_idx_post=0):
    
    end_idx_pre = start_idx_pre + num_pre
    end_idx_post = start_idx_post + num_post
    
    conns = [(i, j, weight, delay) for i in range(start_idx_pre, end_idx_pre) \
                                   for j in range(start_idx_post, end_idx_post)]

    return conns


def one2one(num_neurons, weight=2., delay=1., start_idx=0):

    end_idx = start_idx + num_neurons
    conns = [(i, i, weight, delay) for i in range(start_idx, end_idx)]
    
    return conns


def wta(num_neurons, weight=-2., delay=1., start_idx=0):

    end_idx = start_idx + num_neurons
    conns = [(i, j, weight, delay) for i in range(start_idx, end_idx) \
                                   for j in range(start_idx, end_idx) if i != j]
    
    return conns


def wta_interneuron(num_neurons, ff_weight=2., fb_weight=-2., delay=1., 
                    start_idx=0):
                        
    conn_ff = one2one(num_neurons, np.abs(ff_weight), delay, start_idx)
    
    conn_fb = wta(num_neurons, fb_weight, delay, start_idx)
    
    return conn_ff, conn_fb




######### given neuron id lists do connections

def list_all2all(pre, post, weight=2., delay=1., sd=None):
    scale = 0.5*weight if sd is None else sd
    np.random.seed(np.uint32( time.time()*(10**6) ))
    nw = len(pre)*len(post)
    weights = np.random.normal(loc=weight, scale=scale, size=nw)
    weights = np.abs(weights)
    conns = [(pre[i], post[j], weights[i*len(post) + j], delay) \
                                       for j in range(len(post)) \
                                       for i in range(len(pre)) ]

    return conns


def list_one2one(pre, post, weight=2., delay=1.):
    #smallest list guides the connector
    num_conns = len(pre) if len(pre) < len(post) else len(post)
    
    conns = [(pre[i], post[i], weight, delay) for i in range(num_conns)]
    
    return conns


def list_wta(pop, weight=-2., delay=1.):
    conns = [(i, j, weight, delay) for i in pop \
                                   for j in pop if i != j]
    
    return conns


def list_wta_interneuron(pop, inter, ff_weight=2., fb_weight=-2., delay=1.):
    if len(pop) != len(inter):
        raise Exception("In list_wta_interneuron: lengths of populations not equal")
    
    conn_ff = list_one2one(pop, inter, np.abs(ff_weight), delay)
    npop  = len(pop)
    conn_fb = [(inter[i], pop[j], fb_weight, delay) for i in range(npop) \
                                                    for j in range(npop) if i != j]

    
    return conn_ff, conn_fb
