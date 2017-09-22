#WE SHOULDN'T USE SUCH HIGH FREQUENCIES!!!
# DEFAULT_CFG = {'input_noise_rate': 20.,
#                'exc_noise_rate': 1000.,
#                'inh_noise_rate': 1000.,
#                'w_noise': 0.1,
#                'w_exc': {'prob': 0.1, 'mu': 0.1, 'sigma': 0.1, 'low': 0, 'high': 20.},
#                'd_exc': {'prob': 0.1, 'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
#                'w_inh': {'prob': 0.1, 'mu': 0.5, 'sigma': 0.1, 'low': 0, 'high': 20.},
#                'd_inh': {'prob': 0.1, 'mu': 0.75, 'sigma': 0.375, 'low': 1., 'high': 14.4},
#                'w_in': {'mu': 0.001, 'sigma': 0.01, 'low': 0, 'high': 20.},
#                'd_in': {'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
#                }

DEFAULT_CFG = {'input_noise_rate': 5.,
               'exc_noise_rate': 250.,
               'inh_noise_rate': 250.,
               'w_noise': 0.4,
               'w_exc': {'prob': 0.1, 'mu': 0.4, 'sigma': 0.1, 'low': 0, 'high': 20.},
               'd_exc': {'prob': 0.1, 'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
               'w_inh': {'prob': 0.1, 'mu': 2.0, 'sigma': 0.1, 'low': 0, 'high': 20.},
               'd_inh': {'prob': 0.1, 'mu': 0.75, 'sigma': 0.375, 'low': 1., 'high': 14.4},
               'w_in': {'mu': 0.004, 'sigma': 0.01, 'low': 0, 'high': 20.},
               'd_in': {'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
              }
class RandomBalancedNetwork():

    def __init__(self, sim, n_neurons, proportion=0.8, cfg=DEFAULT_CFG,
                 generate_on_machine=True):
        self.sim = sim
        self.cfg = cfg
        self._n_neurons = n_neurons
        self._proportion = proportion
        self._n_exc = int(proportion*n_neurons)
        self._n_inh = n_neurons - self._n_exc
        self._gen_on_machine = generate_on_machine
        self._build_populations()

        self._build_projections()

        self.output = self._pops['main']['exc']

    def _build_populations(self):
        sim = self.sim
        cfg = self.cfg
        pops = {'main': {}}

        pops['main']['exc'] = sim.Population(
                                self._n_exc, sim.IF_curr_exp, {}, label='excitatory')
        pops['main']['inh'] = sim.Population(
                                self._n_inh, sim.IF_curr_exp, {}, label='inhibitory')
        pops['main']['input'] = sim.Population(self._n_inh,
                                        sim.SpikeSourcePoisson,
                                        {'rate': cfg['input_noise_rate']},
                                        label='input noise')

        # initialise voltages to random numbers
        v_init_distr = sim.RandomDistribution('uniform', parameters=[-65, -55])
        pops['main']['exc'].initialize('v', v_init_distr)
        pops['main']['inh'].initialize('v', v_init_distr)


        pops['noise'] = {}
        pops['noise']['exc'] = sim.Population(
                                    self._n_exc, sim.SpikeSourcePoisson,
                                    {'rate': cfg['exc_noise_rate']},
                                    label='excitatory noise input')
        pops['noise']['inh'] = sim.Population(
                                    self._n_inh, sim.SpikeSourcePoisson,
                                    {'rate': cfg['inh_noise_rate']},
                                    label='inhibitory noise input')

        self._pops = pops

    def _build_projections(self):
        sim = self.sim
        cfg = self.cfg
        gom = self._gen_on_machine
        pops = self._pops


        projs = {}

        #noise inputs
        projs['exc_noise_to_exc'] = sim.Projection(
                                    pops['noise']['exc'], pops['main']['exc'],
                                    sim.OneToOneConnector(
                                        weights=cfg['w_noise'], generate_on_machine=gom),
                                    label='input noise exc to main exc')

        projs['inh_noise_to_inh'] = sim.Projection(
                                    pops['noise']['inh'], pops['main']['inh'],
                                    sim.OneToOneConnector(
                                        weights=cfg['w_noise'], generate_on_machine=gom),
                                    label='input noise inh to main inh')

        #main network
        weights_exc = sim.RandomDistribution('normal',
                         [cfg['w_exc']['mu'], cfg['w_exc']['sigma']])

        delays_exc = sim.RandomDistribution('normal',
                         [cfg['d_exc']['mu'], cfg['d_exc']['sigma']])

        projs['exc_to_exc'] = sim.Projection(
                                    pops['main']['exc'], pops['main']['exc'],
                                    sim.FixedProbabilityConnector(cfg['w_exc']['prob'],
                                        weights=weights_exc, delays=delays_exc,
                                        generate_on_machine=gom),
                                    label='main exc to main exc', target='excitatory')

        projs['exc_to_inh'] = sim.Projection(
                                    pops['main']['exc'], pops['main']['inh'],
                                    sim.FixedProbabilityConnector(cfg['w_exc']['prob'],
                                        weights=weights_exc, delays=delays_exc,
                                        generate_on_machine=gom),
                                    label='main exc to main inh', target='excitatory')


        weights_inh = sim.RandomDistribution('normal',
                                     [cfg['w_inh']['mu'], cfg['w_inh']['sigma']])

        delays_inh = sim.RandomDistribution('normal',
                                     [cfg['d_inh']['mu'], cfg['d_inh']['sigma']])

        projs['inh_to_inh'] = sim.Projection(
                                    pops['main']['inh'], pops['main']['inh'],
                                    sim.FixedProbabilityConnector(cfg['w_inh']['prob'],
                                        weights=weights_inh, delays=delays_inh,
                                        generate_on_machine=gom),
                                    label='main inh to main inh', target='inhibitory')

        projs['inh_to_exc'] = sim.Projection(
                                    pops['main']['inh'], pops['main']['exc'],
                                    sim.FixedProbabilityConnector(cfg['w_inh']['prob'],
                                        weights=weights_inh, delays=delays_inh,
                                        generate_on_machine=gom),
                                    label='main inh to main exc', target='inhibitory')
        
        
        weights_in = sim.RandomDistribution('normal',
                                     [cfg['w_in']['mu'], cfg['w_in']['sigma']])

        delays_in = sim.RandomDistribution('normal',
                                     [cfg['d_in']['mu'], cfg['d_in']['sigma']])
        projs['input_to_exc'] = sim.Projection(
                                    pops['main']['input'], pops['main']['exc'],
                                    sim.AllToAllConnector(
                                                weights=weights_in, delays=delays_in,
                                                generate_on_machine=gom),
                                    label='main input to main exc', target='excitatory')
        
        self._projs = projs


