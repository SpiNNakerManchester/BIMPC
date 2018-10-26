import numpy as np
import matplotlib.pyplot as plt
import spynnaker7.pyNN as sim
from random_balanced_network import RandomBalancedNetwork


CFG = {'input_noise_rate': 5.,
       'exc_noise_rate': 250., 'inh_noise_rate': 250.,'w_noise': 0.32,
       'exc_noise_start': 10., 'inh_noise_start': 0.,
       'w_exc': {'prob': 0.1, 'mu': 0.4, 'sigma': 0.1, 'low': 0, 'high': 20.},
       'd_exc': {'prob': 0.1, 'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
       'w_inh': {'prob': 0.1, 'mu': 2.0, 'sigma': 0.1, 'low': 0, 'high': 20.},
       'd_inh': {'prob': 0.1, 'mu': 0.75, 'sigma': 0.375, 'low': 1., 'high': 14.4},
       'w_in': {'mu': 0.001, 'sigma': 0.01, 'low': 0, 'high': 20.},
       'd_in': {'mu': 1.5, 'sigma': 0.75, 'low': 1., 'high': 14.4},
      }


w2s = 4.8
w_noise = w2s*(1./10.)
w_ctrl = w2s*(1./3.)
w_inh = w2s*0.05
w_recurse = w2s*0.333

sim.setup(timestep=1., min_delay=1., max_delay=32.)

sim.set_number_of_neurons_per_core('IF_curr_exp', 50)

rbn_left = RandomBalancedNetwork(sim, 300, cfg=CFG)
rbn_left.output.record()

rbn_right = RandomBalancedNetwork(sim, 300, cfg=CFG)
rbn_right.output.record()

left_noise = sim.Population(50, sim.SpikeSourcePoisson,
                            {'rate': 10}, label='left noise')

right_noise = sim.Population(50, sim.SpikeSourcePoisson,
                            {'rate': 10}, label='right noise')

left_agent = sim.Population(100, sim.IF_curr_exp, {},
                            label='left agent')
left_agent.record()

right_agent = sim.Population(100, sim.IF_curr_exp, {},
                             label='right agent')
right_agent.record()

left_control = sim.Population(2, sim.IF_curr_exp, {},
                              label='left control')

right_control = sim.Population(2, sim.IF_curr_exp, {},
                               label='right control')

left_control.record()
right_control.record()

sim.Projection(rbn_left.output, left_agent,
               sim.FixedProbabilityConnector(0.1, weights=w_noise),
               label='left rbn to agent')

sim.Projection(rbn_right.output, right_agent,
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

# sim.Projection(left_agent, right_agent,
#                sim.FixedProbabilityConnector(0.5, weights=w2s),
#                target='inhibitory',
#                label='left to right inhibition')
#
# sim.Projection(right_agent, left_agent,
#                sim.FixedProbabilityConnector(0.5, weights=w2s),
#                target='inhibitory',
#                label='right to left inhibition')

sim.Projection(left_control, right_control,
               sim.OneToOneConnector(weights=w2s),
               target='inhibitory',
               label='left to right control inhibition')

sim.Projection(right_control, left_control,
               sim.OneToOneConnector(weights=w2s),
               target='inhibitory',
               label='right to left control inhibition')

# keep input current comming into decision
sim.Projection(left_control, left_control,
               sim.OneToOneConnector(weights=w_recurse),
               label='recursive decision left')

sim.Projection(right_control, right_control,
               sim.OneToOneConnector(weights=w_recurse),
               label='recursive decision left')

sim.run(3000)

spikes_left = np.array(left_control.getSpikes(compatible_output=True))
spikes_right = np.array(right_control.getSpikes(compatible_output=True))
spikes_rbn_left = np.array(rbn_left.output.getSpikes(compatible_output=True))
spikes_rbn_right = np.array(rbn_right.output.getSpikes(compatible_output=True))
spikes_agent_left = np.array(left_agent.getSpikes(compatible_output=True))
spikes_agent_right = np.array(right_agent.getSpikes(compatible_output=True))
sim.end()

plt.figure()
ax = plt.subplot(3, 2, 1)
plt.plot(spikes_left[:, 1], spikes_left[:, 0], 'xb', markersize=5, label='left')
plt.margins(0.1, 0.1)
plt.grid()
plt.legend()

ax = plt.subplot(3, 2, 2)
plt.plot(spikes_right[:, 1], spikes_right[:, 0], '+r', markersize=5, label='right')
plt.margins(0.1, 0.1)
plt.grid()
plt.legend()

ax = plt.subplot(3, 2, 3)
plt.plot(spikes_rbn_left[:, 1], spikes_rbn_left[:, 0],
         '.b', markersize=5, label='rbn left')
plt.margins(0.1, 0.1)
plt.grid()
plt.legend()

ax = plt.subplot(3, 2, 4)
plt.plot(spikes_rbn_right[:, 1], spikes_rbn_right[:, 0],
         '.r', markersize=5, label='rbn right')
plt.margins(0.1, 0.1)
plt.grid()
plt.legend()

ax = plt.subplot(3, 2, 5)
plt.plot(spikes_agent_left[:, 1], spikes_agent_left[:, 0],
         '.b', markersize=5, label='agent left')
plt.margins(0.1, 0.1)
plt.grid()
plt.legend()

ax = plt.subplot(3, 2, 6)
plt.plot(spikes_agent_right[:, 1], spikes_agent_right[:, 0],
         '.r', markersize=5, label='agent right')
plt.margins(0.1, 0.1)
plt.grid()
plt.legend()

plt.show()




