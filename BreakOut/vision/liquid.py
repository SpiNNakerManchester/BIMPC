from sim_tools.common import *
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std
                                 
import time

class Liquid():
    
    def __init__(self, sim, input_cells, neurons_per_simple, learning_on,
                 in_width, in_height, coords, simples_per_liquid, 
                 neurons_per_liquid, cfg):
        
        self.sim = sim
        self.cfg = cfg
        
        self.in_pops = input_cells
        
        self.neurons_per_simple = neurons_per_simple
        self.simple = simple_cells
        
        self.learning_on = learning_on
        self.coords = coords
        self.num_neurons = neurons_per_liquid
        self.num_exc = cfg['exc_per_cent']*self.num_neurons
        self.num_inh = self.num_neurons - num_exc
        self.num_conns = self.num_neurons*cfg['connection_probability']
        
        self.img_w = in_width
        self.img_h = in_height
        
        self.build_connectors()
        self.build_populations()
        self.build_projections()
        self.connect_input()



    def get_synapse_dynamics(self):
        if not self.learning_on:
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
        
        
    # try hub connectivity in 
    # http://www.sciencedirect.com/science/article/pii/S0957417411009523
    def build_connectors(self):
        sim = self.sim
        cfg = self.cfg
        max_delay = cfg['max_delay']
        max_weight = cfg['max_weight']
        num_exc = self.num_exc
        e2e_conns = []
        e2i_conns = []
        
        for pre in range(self.num_exc):
            post = np.random.choice(self.num_neurons, 
                                    self.num_conns, replace=False)
            exc = post[np.where(post < num_exc)]
            inh = post[np.where(post >= num_exc)] - num_exc
            
            np.random.seed(np.uint32(time.time()*(10*6)))
            delays = np.random.choice(max_delay, len(exc), replace=True)
            weights = np.random.normal(max_weight, scale=0.01*max_weight, \
                                       size=len(exc))
            e2e_conns += [(pre, exc[i], weights[i], delays[i]) \
                          for i in len(exc) if i != pre]

            np.random.seed(np.uint32(time.time()*(10*6)))
            delays = np.random.choice(max_delay, len(inh), replace=True)
            weights = np.random.normal(max_weight, scale=0.01*max_weight, \
                                       size=len(inh))
            e2i_conns += [(pre, inh[i], weights[i], delays[i]) for i in len(inh)]


        for pre in range(self.num_inh):
            post = np.random.choice(self.num_neurons, 
                                    self.num_conns, replace=False)
            exc = post[np.where(post < num_exc)]
            # inh = post[np.where(post >= num_exc)] - num_exc
            
            np.random.seed(np.uint32(time.time()*(10*6)))
            weights = -np.random.normal(max_weight, scale=0.01*max_weight, \
                                        size=len(exc))
            i2e_conns += [(pre, exc[i], weights[i], 1.) for i in len(exc)]

            # np.random.seed(np.uint32(time.time()*(10*6)))
            # weights = -np.random.normal(max_weight, scale=0.01*max_weight, \
                                        # size=len(inh))
            # i2i_conns += [(pre, inh[i], weights[i], 1.) for i in len(inh) \
                          # if i != pre]

        self.e2e_conns = e2e_conns
        self.e2i_conns = e2i_conns
        self.i2e_conns = i2e_conns
#        self.i2i_conns = i2i_conns



    def build_populations(self):
        
        def loc2lbl(loc, pop):
            return "liquid (%d, %d) - %s"%(loc[ROW], loc[COL], pop)

        sim = self.sim
        cfg = self.cfg
        
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']
        
        pops = {}
        # excitatory neurons
        pops['exc'] = sim.Population(self.num_exc, exc_cell, exc_parm,
                                     label=loc2lbl(self.coords, \
                                                   'excitatory') )
        # randomize population parameters
        for p in exc_parm:
            rng = sim.random.NumpyRNG( seed=np.uint32(time.time()*(10**6)) )
            rd = sim.random.RandomDistribution('normal', mu=exc_parm[p], 
                                               sigma=0.01*exc_parm[p], 
                                               rng=rng)
            pops['exc'].rset(p, rd)

        # inhibitory neurons
        pops['inh'] = sim.Population(self.num_inh, inh_cell, inh_parm,
                                     label=loc2lbl(self.coords, \
                                                   'inhibitory') )
        # randomize population parameters
        for p in inh_parm:
            rng = sim.random.NumpyRNG( seed=np.uint32(time.time()*(10**6)) )
            rd = sim.random.RandomDistribution('normal', mu=exc_parm[p], 
                                               sigma=0.01*exc_parm[p], 
                                               rng=rng)
            pops['inh'].rset(p, rd)

        
        if cfg['record']['spikes']:
            for k in pops:
                pops[k].record()

        if cfg['record']['voltages']:
            for k in pops:
                pops[k].record_v()
        
        self.pops = pops



    def build_projections(self):
        def loc2lbl(loc, pop):
            return "liquid (%d, %d) - %s"%(loc[ROW], loc[COL], pop)
            
        sim = self.sim
        cfg = self.cfg

        projs = {}
        syn_dyn = self.get_synapse_dynamics()
        
        e2e = sim.FromListConnector(self.e2e_conns)
        projs['e2e'] = sim.Projection(self.pops['exc'], self.pops['exc'], \
                                      e2e, target='excitatory', \
                                      synapse_dynamics=syn_dyn, \
                                      label=loc2lbl(self.coords, 'e2e'))

        e2i = sim.FromListConnector(self.e2i_conns)
        projs['e2i'] = sim.Projection(self.pops['exc'], self.pops['inh'], \
                                      e2i, target='excitatory', \
                                      synapse_dynamics=syn_dyn, \
                                      label=loc2lbl(self.coords, 'e2i'))

        i2e = sim.FromListConnector(self.i2e_conns)
        projs['i2e'] = sim.Projection(self.pops['inh'], self.pops['exc'], \
                                      e2e, target='inhibitory', \
                                      # synapse_dynamics=syn_dyn, \
                                      label=loc2lbl(self.coords, 'i2e'))

        self.projs = projs



    def connect_input(self):
        sim = self.sim
        cfg = self.cfg
        in2exc_p = cfg['in_to_liquid_exc_probability']
        in2inh_p = cfg['in_to_liquid_exc_probability']
        
        syn_dyn = self.get_synapse_dynamics()

        in_projs = []
        for in_pop in self.in_pops:
            proj = sim.Projection(in_pop, self.pops['exc'],
                                  sim.FixedProbabilityConnector(in2exc_p), \
                                  # synapse_dynamics=syn_dyn, \
                                  label="in to exc %s"%self.coords)
            in_projs.append(proj)

            proj = sim.Projection(in_pop, self.pops['inh'],
                                  sim.FixedProbabilityConnector(in2inh_p), \
                                  # synapse_dynamics=syn_dyn, \
                                  label="in to inh %s"%self.coords)
            in_projs.append(proj)

            
