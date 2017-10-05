import numpy as np
import matplotlib.pyplot as plt
import spynnaker7.pyNN as sim


w2s = 10.
w_keep_alive = w2s*0.84
d_keep_alive = 4
sim.setup(timestep=1., min_delay=1., max_delay=32.)

sim.set_number_of_neurons_per_core('IF_curr_exp', 50)

stim = sim.Population(1, sim.SpikeSourceArray,
                      {'spike_times': [[10]]})
control = sim.Population(1, sim.IF_curr_exp, {'tau_syn_E': 1.},
                         label='control')
control.record()

sim.Projection(stim, control,
               sim.OneToOneConnector(weights=w2s),
               target='excitatory',
               label='stim to control')

sim.Projection(control, control,
               sim.OneToOneConnector(weights=w_keep_alive,
                                     delays=d_keep_alive),
               target='excitatory',
               label='keep alive')

sim.run(1000)

spikes = np.array(control.getSpikes(compatible_output=True))
sim.end()

plt.figure()
ax = plt.subplot(1, 1, 1)
plt.plot(spikes[:, 1], spikes[:, 0], 'xb', markersize=5)
plt.margins(0.1, 0.1)
plt.grid()
plt.show()




