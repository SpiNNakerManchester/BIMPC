import numpy as np
import matplotlib.pyplot as plt
import spynnaker7.pyNN as sim
from random_balanced_network import RandomBalancedNetwork


CFG = {'input_noise_rate': 5.,
       'exc_noise_rate': 250., 'inh_noise_rate': 250.,'w_noise': 0.35,
       'w_exc': {'prob': 0.1, 'mu': 0.4, 'sigma': 0.1, 'low': 0, 'high': 20.},
       'd_exc': {'prob': 0.1, 'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
       'w_inh': {'prob': 0.1, 'mu': 2.0, 'sigma': 0.1, 'low': 0, 'high': 20.},
       'd_inh': {'prob': 0.1, 'mu': 0.75, 'sigma': 0.375, 'low': 1., 'high': 14.4},
       'w_in': {'mu': 0.004, 'sigma': 0.01, 'low': 0, 'high': 20.},
       'd_in': {'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
      }


w2s = 4.8
w_noise = w2s*(1./10.)
w_ctrl = w2s*(1./25.)
w_inh = w2s*0.05

sim.setup(timestep=1., min_delay=1., max_delay=32.)

sim.set_number_of_neurons_per_core('IF_curr_exp', 50)

rbn0 = RandomBalancedNetwork(sim, 400, cfg=CFG)

# rbn0.output.record()

rbn1 = RandomBalancedNetwork(sim, 450, cfg=CFG)

# rbn1.output.record()

left_noise = sim.Population(50, sim.SpikeSourcePoisson,
                            {'rate': 10}, label='left noise')

right_noise = sim.Population(50, sim.SpikeSourcePoisson,
                            {'rate': 10}, label='right noise')

left_agent = sim.Population(100, sim.IF_curr_exp, {},
                            label='left agent')

right_agent = sim.Population(100, sim.IF_curr_exp, {},
                             label='right agent')

left_control = sim.Population(2, sim.IF_curr_exp, {},
                              label='left control')

right_control = sim.Population(2, sim.IF_curr_exp, {},
                               label='right control')

left_control.record()
right_control.record()

sim.Projection(rbn0.output, left_agent,
               sim.FixedProbabilityConnector(0.1, weights=w_noise),
               label='left rbn to agent')

sim.Projection(rbn1.output, right_agent,
               sim.FixedProbabilityConnector(0.1, weights=w_noise),
               label='right rbn to agent')

sim.Projection(left_noise, left_agent,
               sim.FixedProbabilityConnector(0.1, weights=w_noise),
               label='left poisson to agent')

sim.Projection(right_noise, right_agent,
               sim.FixedProbabilityConnector(0.1, weights=w_noise),
               label='right poisson to agent')

sim.Projection(left_agent, left_control,
               sim.AllToAllConnector(weights=w_ctrl,
                                     allow_self_connections=True),
               label='left agent to control')

sim.Projection(right_agent, right_control,
               sim.AllToAllConnector(weights=w_ctrl,
                                     allow_self_connections=True),
               label='right agent to control')

sim.Projection(left_agent, right_agent,
               sim.FixedProbabilityConnector(0.5, weights=w2s),
               target='inhibitory',
               label='left to right inhibition')

sim.Projection(right_agent, left_agent,
               sim.FixedProbabilityConnector(0.5, weights=w2s),
               target='inhibitory',
               label='right to left inhibition')

sim.Projection(left_control, right_control,
               sim.OneToOneConnector(weights=w2s),
               target='inhibitory',
               label='left to right control inhibition')

sim.Projection(right_control, left_control,
               sim.OneToOneConnector(weights=w2s),
               target='inhibitory',
               label='right to left control inhibition')

sim.run(10000)

spikes0 = np.array(left_control.getSpikes(compatible_output=True))
spikes1 = np.array(right_control.getSpikes(compatible_output=True))

sim.end()

plt.figure()
plt.plot(spikes0[:, 1], spikes0[:, 0], 'xb', markersize=2, label='left')
plt.plot(spikes1[:, 1], spikes1[:, 0], '+r', markersize=2, label='right')
plt.margins(0.1, 0.1)
plt.grid()
plt.legend()
plt.show()




