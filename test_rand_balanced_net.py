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

sim.setup(timestep=1., min_delay=1., max_delay=32.)

sim.set_number_of_neurons_per_core('IF_curr_exp', 50)

rbn0 = RandomBalancedNetwork(sim, 400, cfg=CFG)

rbn0.output.record()

rbn1 = RandomBalancedNetwork(sim, 450, cfg=CFG)

rbn1.output.record()


sim.run(1000)

spikes0 = np.array(rbn0.output.getSpikes(compatible_output=True))
spikes1 = np.array(rbn1.output.getSpikes(compatible_output=True))

sim.end()

plt.figure()
plt.plot(spikes0[:, 1], spikes0[:, 0], 'xb', markersize=2)
plt.plot(spikes1[:, 1], spikes1[:, 0], '+r', markersize=2)
plt.show()




