import numpy as np
import matplotlib.pyplot as plt
import spynnaker7.pyNN as sim


w2s = 5.
w_keep_alive = w2s
w_control_inh = w2s*0.25
w_inh_feedback = w2s*2.
d_keep_alive = 2
d_inh_feedback = 30
d_inh_feedforward = 30
sim.setup(timestep=1., min_delay=1., max_delay=32.)

sim.set_number_of_neurons_per_core('IF_curr_exp', 50)

stim = sim.Population(1, sim.SpikeSourceArray,
                      {'spike_times': [[10]]})
control = sim.Population(1, sim.IF_curr_exp,
                         {'tau_refrac': 1.},
                         label='control')
control.record()

inh_ctrl = sim.Population(1, sim.IF_curr_exp, {},
                          label='inh control')


sim.Projection(stim, control,
               sim.OneToOneConnector(weights=w2s),
               target='excitatory',
               label='stim to control')

sim.Projection(control, control,
               sim.OneToOneConnector(weights=w_keep_alive,
                                     delays=d_keep_alive),
               target='excitatory',
               label='keep alive')

sim.Projection(control, inh_ctrl,
               sim.OneToOneConnector(weights=w_control_inh,
                                     delays=d_inh_feedforward),
               target='excitatory',
               label='control to inh')

sim.Projection(inh_ctrl, control,
               sim.OneToOneConnector(weights=w_inh_feedback,
                                     delays=d_inh_feedback),
               target='inhibitory',
               label='control to inh')

sim.run(1000)

spikes = np.array(control.getSpikes(compatible_output=True))
sim.end()

plt.figure()
ax = plt.subplot(1, 1, 1)
plt.plot(spikes[:, 1], spikes[:, 0], 'xb', markersize=5)
plt.margins(0.1, 0.1)
plt.grid()
plt.show()




