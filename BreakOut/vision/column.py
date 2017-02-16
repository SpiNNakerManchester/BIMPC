from sim_tools.common import *
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std




class V1MultiColumn():
    
    def __init__(self, sim, lgn, learning_on, in_width, in_height, 
                 in_location, group_size, cfg):
        # print("\t\tBuilding MultiColumn ... ")
        self.cfg = cfg
        self.sim = sim
        self.learn_on = learning_on
        
        self.lgn = lgn
        self.retina = lgn.retina
        self.width   = lgn.width 
        self.width2  = lgn.width2
        self.width4  = lgn.width4
        self.height  = lgn.height
        self.height2 = lgn.height2
        self.height4 = lgn.height4
        self.in_weight_func = cfg['col_weight_func']
        self.in_location = in_location
        self.group_size = group_size

        
        self.pix_key   = 'cs'
        self.feat_keys = [k for k in lgn.pops.keys() if k != 'cs']
        self.num_in_ctx = len(self.feat_keys)
        
        # print("\t\t\tbuilding input indices...")
        self.build_input_indices_weights()
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
        
    def get_map_params(self, k):
        if k == 'cs4':
            kk = 'ctr_srr_quarter'
            width     = self.retina.filter_width4
            height    = self.retina.filter_height4
        elif k == 'cs2':
            kk = 'ctr_srr_half'
            width     = self.retina.filter_width2
            height    = self.retina.filter_height2
        else:
            kk = 'ctr_srr'
            width     = self.retina.filter_width
            height    = self.retina.filter_height
            
        step      = self.retina.cfg[kk]['step']
        start     = self.retina.cfg[kk]['start']
        krn_width = self.retina.cfg[kk]['width']        

        return step, start, width, height, krn_width
    
    def get_row_col_limits(self, half_krn_width):
        #location is in highest resolution scale (i.e. 'ctr_srr')
        hkw = half_krn_width
        fr_r = max(0, self.in_location[ROW] - hkw)
        to_r = min(self.height, self.in_location[ROW] + hkw + 1)
        fr_c = max(0, self.in_location[COL] - hkw)
        to_c = min(self.width,  self.in_location[COL] + hkw + 1)
        
        return [fr_r, fr_c], [to_r, to_c]

    def build_input_indices_weights(self):
        cfg = self.cfg
        indices = {k: [] for k in self.lgn.output_keys()}
        weights = {k: [] for k in self.lgn.output_keys()}
        step    = 1; start   = 0; width   = 1; height  = 1
        sanity = {}; frm = []; to = []
        ssmp_r = 0; ssmp_c = 0
        my_r = self.in_location[ROW]; my_c = self.in_location[COL]
        to_delete = []
        half_rec_w = cfg['in_receptive_width']//2
        for k in indices:
            # print("----------- KEY %s ---------------"%k)
            step, start, width, height, krn_width = self.get_map_params(k)
            half_krn_w = max(krn_width//2, half_rec_w)
            frm[:], to[:] = self.get_row_col_limits(half_krn_w)
            sanity.clear()
            for r in range(frm[ROW], to[ROW]):
                ssmp_r = (r - start)//step
                if ssmp_r not in sanity:
                    sanity[ssmp_r] = []
                for c in range(frm[COL], to[COL]):
                    ssmp_c = (c - start)//step
                    if ssmp_c in sanity[ssmp_r]:
                        continue
                        
                    d = np.sqrt( (my_r - r)**2 + (my_c - c)**2 )
                    w = self.in_weight_func(d)

                    if w < cfg['min_scale_weight']:
                        # print("dist %s, weight %s"%(d, w))
                        continue

                    sanity[ssmp_r].append(ssmp_c)
                    indices[k].append( int(ssmp_r*width + ssmp_c) )
                    weights[k].append( w )
            
            if len(indices[k]) == 0:
                to_delete.append(k)

        for k in to_delete:
            # print("(%d, %d) %s"%(my_r, my_c, k))
            del indices[k]
            del weights[k]

        self.in_indices = indices
        self.in_weights = weights
        
    
    def build_connectors(self):
        conns = {}
        cfg = self.cfg
        size = self.group_size
        sipl_idx = [i for i in range(size)]
        
        for k in self.in_indices:
            in_idx = self.in_indices[k]
            in_ws  = self.in_weights[k]
            conns[k] = conn_std.list_all2all(in_idx, sipl_idx, 
                                             weight=cfg['pix_in_weight'], 
                                             delay=2., sd=0.1,
                                             in_weight_scaling=in_ws)

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
            pops['simple'].record()
            pops['wta_inh'].record()

        if cfg['record']['voltages']:
            pops['simple'].record_v()
            pops['wta_inh'].record_v()
        
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
        projs = {}
        
        for k in self.conns:
            if k == 'sipl2intr':
                continue
            
            #both channels land on the same neuron group
            
            in_pop = self.lgn.pops['on'][k]['output']
            conn = sim.FromListConnector(self.conns[k])
            syn_dyn = self.get_synapse_dynamics()
            projs['in2sipl'] = sim.Projection(in_pop, self.pops['simple'],
                                              conn, synapse_dynamics=syn_dyn,
                                              label='input to simple')

            in_pop = self.lgn.pops['off'][k]['output']
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
        
        
        
