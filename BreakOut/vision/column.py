from sim_tools.common import *
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std




class V1MultiColumn():
    
    def __init__(self, sim, lgn, learning_on,
                 in_width, in_height, in_location, in_receptive_width, 
                 group_size, cfg):
        # print("\t\tBuilding MultiColumn ... ")
        self.cfg = cfg
        self.sim = sim
        self.learn_on = learning_on
        
        self.lgn = lgn
        self.in_width  = in_width
        self.in_height = in_height
        self.in_location = in_location
        self.in_receptive_width = in_receptive_width
        self.group_size = group_size
        self.num_in_neruons = in_receptive_width*in_receptive_width
        
        self.pix_key   = 'cs'
        self.feat_keys = [k for k in lgn.pops.keys() if k != 'cs']
        self.num_in_ctx = len(self.feat_keys)
        
        # print("\t\t\tbuilding input indices...")
        self.build_input_indices()
        # print("\t\t\tdone!")
        
        # print("\t\t\tbuilding connectors...")
        self.build_connectors()
        # print("\t\t\tdone!")
        
        # print("\t\t\tbuilding populations...")
        self.build_populations()
        # print("\t\t\tdone!")
        
        self.build_projections()
        
        

    def update_weights(self, new_weights):
        pass
        
        
    def build_input_indices(self):
        indices = []
        hlf_in_w = self.in_receptive_width//2
        fr_r = max(0, self.in_location[ROW] - hlf_in_w)
        to_r = min(self.in_height, self.in_location[ROW] + hlf_in_w + 1)
        fr_c = max(0, self.in_location[COL] - hlf_in_w)
        to_c = min(self.in_width,  self.in_location[COL] + hlf_in_w + 1)
        
        for r in range(fr_r, to_r):
            for c in range(fr_c, to_c):
                indices.append( int(r*self.in_width + c) )
        
        self.in_indices = indices
    
        
    
    def build_connectors(self):
        conns = {}
        cfg = self.cfg
        size = self.group_size
        in_idx = self.in_indices
        sipl_idx = [i for i in range(size)]
        num_in = self.num_in_neruons
        num_in_ctx = self.num_in_ctx
        
        conns['in2sipl'] = conn_std.list_all2all(in_idx, sipl_idx, 
                                            weight=cfg['pix_in_weight'], 
                                            delay=1., sd=0.01)
        

        conns['sipl2intr'] = conn_std.list_wta_interneuron(sipl_idx, sipl_idx, 
                                                    ff_weight=cfg['w2s'], 
                                                    fb_weight=-cfg['w2s'], 
                                                    delay=1.)
        
        self.conns = conns


    def build_populations(self):
        def loc2lbl(loc, pop):
            return "column (%d, %d) - %s"%(loc[ROW], loc[COL], pop)

        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['wta_inh_cell']['cell'], None)
        inh_parm = cfg['wta_inh_cell']['params']
        
        pops = {}
        pops['simple'] = sim.Population(self.group_size,
                                        exc_cell, exc_parm,
                                        label=loc2lbl(self.in_location, \
                                                     'simple') )

        pops['wta_inh'] = sim.Population(self.group_size,
                                         inh_cell, inh_parm,
                                         label=loc2lbl(self.in_location, 'wta') )
                                                
        
        if cfg['record']['spikes']:
            for k in pops:
                pops[k].record()

        if cfg['record']['voltages']:
            for k in pops:
                pops[k].record_v()
        
        self.pops = pops
    
    
    def get_synapse_dynamics(self):
        if not self.learn_on:
            return None
            
        cfg = self.cfg['stdp']
        sim = self.sim
        
        stdp_model = sim.STDPMechanism(
            timing_dependence = sim.SpikePairRule(tau_plus=cfg['tau_plus'], 
                                                  tau_minus=cfg['tau_minus']),
            weight_dependence = sim.AdditiveWeightDependence(w_min=cfg['w_min'], 
                                                             w_max=cfg['w_max'], 
                                                             A_plus=cfg['a_plus'], 
                                                             A_minus=cfg['a_minus']),
        )
        syn_dyn = sim.SynapseDynamics(slow=stdp_model)
        
        return syn_dyn
    
    
    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        in_pop = self.lgn.pops['cs']['output']
        
        projs = {}
        
        conn = sim.FromListConnector(self.conns['in2sipl'])
        syn_dyn = self.get_synapse_dynamics()
        projs['in2sipl'] = sim.Projection(in_pop, self.pops['simple'],
                                          conn, synapse_dynamics=syn_dyn,
                                          label='input to simple')
        
        conn = sim.FromListConnector(self.conns['sipl2intr'][0])
        projs['sipl2intr'] = sim.Projection(self.pops['simple'],
                                            self.pops['wta_inh'],
                                            conn, 
                                            label='simple to inter')

        conn = sim.FromListConnector(self.conns['sipl2intr'][1])
        projs['intr2sipl'] = sim.Projection(self.pops['wta_inh'],
                                            self.pops['simple'],
                                            conn, 
                                            label='inter to simple')
            
        self.projs = projs





    def get_weights_input(self):
        sp = self.projs['in2sipl']
        all_ws = sp.getWeights(format='array', gather=False)
        # print(all_ws[self.in_indices, 0].shape)
        weights = [ all_ws[self.in_indices, i] for i in range(self.group_size) ]
        
        return weights
        
        
        
    def get_weights_pictures(self, weights):
        w = weights
        recept_shape = (self.in_receptive_width, self.in_receptive_width)
        
        imgs = [ np.array(w[i]).reshape(recept_shape) for i in range(len(w)) ]
        
        return imgs
        
        
        
