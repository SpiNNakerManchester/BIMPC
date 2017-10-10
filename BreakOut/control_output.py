W2S = 5.
DEFAULT_CONFIG = dict(w2s=W2S,
                      w_keep_alive = W2S,
                      w_control_inh = W2S*0.25,
                      w_inh_feedback = W2S*2.,
                      d_keep_alive = 2,
                      d_inh_feedback = 20,
                      d_inh_feedforward = 20,
                      tau_refrac = 2.,
                      label='control')

class ControlOutput():

    def __init__(self, sim, num_neurons, cfg=DEFAULT_CONFIG):
        self.sim = sim
        self.num_neurons = num_neurons
        self.cfg = cfg
        self.input = None
        self.output = None
        self.pops = None
        self.projs = None

        self.build_populations()
        self.build_projections()

    def build_populations(self):
        sim = self.sim
        cfg = self.cfg
        pops = {}

        pops['main'] = sim.Population(self.num_neurons,
                            sim.IF_curr_exp, {'tau_refrac': cfg['tau_refrac']},
                            label="{} main".format(cfg['label']))

        pops['inh'] = sim.Population(self.num_neurons,
                            sim.IF_curr_exp, {'tau_refrac': 1.},
                            label="{} self inh".format(cfg['label']))

        self.pops = pops

        self.input = self.pops['main']
        self.output = self.pops['main']

    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        projs = {}

        projs['main2main'] = sim.Projection(
                                self.pops['main'], self.pops['main'],
                                sim.OneToOneConnector(
                                    weights=cfg['w_keep_alive'],
                                    delays=cfg['d_keep_alive']),
                                target='excitatory',
                                label='{} keep alive'.format(cfg['label']))

        projs['main2inh'] = sim.Projection(
                                self.pops['main'], self.pops['inh'],
                                sim.OneToOneConnector(
                                    weights=cfg['w_control_inh'],
                                    delays=cfg['d_inh_feedforward']),
                                target='excitatory',
                                label='control to inh')

        projs['inh2main'] = sim.Projection(
                                self.pops['inh'], self.pops['main'],
                                sim.OneToOneConnector(
                                    weights=cfg['w_inh_feedback'],
                                    delays=cfg['d_inh_feedback']),
                                target='inhibitory',
                                label='{} control to inh'.format(cfg['label']))

        self.projs = projs