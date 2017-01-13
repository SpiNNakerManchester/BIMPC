from sim_tools.common import *
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std

class Readout():

    def __init__(self, sim, liquid, neurons_per_liquid, learning_on,
                 in_width, in_height, coords, simples_per_liquid, 
                 neurons_per_liquid, cfg):



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
