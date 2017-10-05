"""
Simple test for neuromodulated-STDP

We take 10 populations of 5 stimuli neurons and connect to each
10 post-synaptic neurons. The spiking of stimuli causes some
spikes in post-synaptic neurons initially.

We then inject reward signals from dopaminergic neurons
periodically to reinforce synapses that are active. This
is followed by increased weights of some synapses and thus
increased response to the stimuli.

We then proceed to inject punishment signals from dopaminergic
neurons which causes an inverse effect to reduce response of
post-synaptic neurons to the same stimuli.

"""

import spynnaker7.pyNN as sim
import pylab
import numpy as np
import pylab




def switch(i):
    return (True if i else False)

punish  = switch(0)
reward  = switch(1 and not punish)
causal  = switch(1)
stdp    = switch(1)
spingen = switch(1)

num_neurons = 1
timestep = 1.0
stim_rate = 10
duration = 2000
plastic_weights = 4.

# Times of rewards and punishments
# rewards = [x for x in range(2000, 2010)] + \
          # [x for x in range(3000, 3020)] + \
          # [x for x in range(4000, 4100)]
# punishments = [x for x in range(5000, 5020)] + \
              # [x for x in range(7000, 7030)] + \
              # [x for x in range(8000, 8020)] + \
              # [x for x in range(9000, 9030)]

tau_plus = 10.
tau_minus = 10.
tau_c = 500.
tau_d = 100.
a_plus = 1.
a_minus = 1.
w_min = 0.
w_max = 10.
w_reward = 0.001
w_punish = 0.001
w_post_stim = 6.

rewards = [53]
punishments = [53]


if causal:
    sources = [45, 1992, 1994]
    targets = [50]
else:
    sources = [50, 1992, 1994]
    targets = [45]#, 100]

cell_params = { 'cm': 0.3,         'i_offset': 0.0,  'tau_m': 10.0,
               'tau_refrac': 2.0, 'tau_syn_E': 1.,  'tau_syn_I': 1.,
               'v_reset': -70.0,  'v_rest': -65.0,  'v_thresh': -55.4
              }

sim.setup(timestep=timestep)

# Create a population of dopaminergic neurons for reward and punishment
if reward:
    reward_pop = sim.Population(1, sim.SpikeSourceArray,
                                {'spike_times' : rewards}, label='reward')
    reward_pop.record()

if punish:
    punish_pop = sim.Population(1, sim.SpikeSourceArray,
                                {'spike_times' : punishments}, label='punishment')
    punish_pop.record()

pre_pops = []
stimulation = []
post_pops = []
reward_projections = []
punishment_projections = []
plastic_projections = []
stim_projections = []


source =sim.Population(num_neurons, sim.SpikeSourceArray,
                       {'spike_times': sources}, label="pre")
source.record()

target_stim = sim.Population(num_neurons, sim.SpikeSourceArray,
                             {'spike_times': targets}, label='post stimulation')
target_stim.record()

target = sim.Population(1, sim.IF_curr_exp_supervision,
                        cell_params, label='post')
target.record()


sim.Projection(target_stim, target, 
               sim.AllToAllConnector(weights=w_post_stim,
                                     generate_on_machine=spingen),
               label='post stimulation synapses')

if reward:
    sim.Projection(reward_pop, target,
                   sim.AllToAllConnector(weights=w_reward,
                                         generate_on_machine=spingen),
                   target='reward', label='reward synapses')

if punish:
    sim.Projection(punish_pop, target,
                   sim.AllToAllConnector(weights=w_punish,
                                         generate_on_machine=spingen),
                   target='punishment', label='punishment synapses')

# Create synapse dynamics with neuromodulated STDP.
if stdp:
    synapse_dynamics = sim.SynapseDynamics(slow=sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus, tau_minus=tau_minus,
            tau_c=tau_c, tau_d=tau_d),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max,
            A_plus=a_plus, A_minus=a_minus
            ),
        neuromodulation=True))
else:
    synapse_dynamics = None

# Create plastic connections between stimulation populations and observed
# neurons
learn_syn = sim.Projection(source, target,
                           sim.AllToAllConnector(weights=plastic_weights,
                                                 generate_on_machine=spingen),
                           synapse_dynamics=synapse_dynamics,
                           target='excitatory', label='Pre-post projection')


sim.run(duration)

#grab spikes
target_spikes = target.getSpikes(compatible_output=True)

source_spikes = source.getSpikes(compatible_output=True)

if reward:
    reward_spikes = reward_pop.getSpikes(compatible_output=True)

if punish:
    punish_spikes = punish_pop.getSpikes(compatible_output=True)

if stdp:
    weights = learn_syn.getWeights(format='array')

sim.end()

if reward:
    print("Reward spikes")
    print(reward_spikes)

if punish:
    print("Punishment spikes")
    print(punish_spikes)


##################################################################################

pylab.figure(figsize=(13,5))

if reward:
    spk = np.array(reward_spikes)
    pylab.plot(spk[:, 1], 0.5*np.ones_like(spk[:, 0]), 'g^')
    for s in spk:
        t = s[1]
        pylab.plot([t, t], [0, num_neurons+1], 'g')
    
if punish:
    spk = np.array(punish_spikes)
    pylab.plot(spk[:, 1], 0.5*np.ones_like(spk[:, 0]), 'r^')
    for s in spk:
        t = s[1]
        pylab.plot([t, t], [0, num_neurons+1], 'r')

markersize=6
spk = np.array(source_spikes)
print("\nSource SPIKES")
print(spk)
pylab.plot(spk[:, 1], spk[:, 0], 'c.', markersize=markersize)

spk = np.array(target_spikes)
print("\nTarget SPIKES")
print(spk)
pylab.plot(spk[:, 1], spk[:, 0] + num_neurons, "b.", markersize=markersize)

pylab.xlabel('Time (ms)')
pylab.ylabel('Neuron Id')
pylab.margins(0.1, 0.1)
pylab.grid()
# pylab.title(title)
        
pylab.savefig("neuromodulation_spikes.pdf")
# pylab.show()

if stdp:
    print("Weights(Initial %s)" % plastic_weights)
    for x in weights:
        print x


    new_w = np.array(weights)#.reshape((-1, 1))
    old_w = plastic_weights*np.ones_like(new_w)
    diff  = new_w - old_w
    print(diff)
    dif_p = (diff > 0)*diff
    dif_n = (diff < 0)*diff

    dif_c = np.zeros((new_w.shape[0], new_w.shape[1], 4))
    dif_c[:, :, 0] = dif_n
    dif_c[:, :, 1] = dif_p

    pylab.figure()
    ax = pylab.subplot(1, 3, 1)
    pylab.imshow(old_w, cmap='Greys_r', interpolation='none')

    ax = pylab.subplot(1, 3, 2)
    pylab.imshow(new_w, cmap='Greys_r', interpolation='none')

    ax = pylab.subplot(1, 3, 3)
    pylab.imshow(dif_c, interpolation='none')

    pylab.savefig("modulated_weight_change.pdf")

# pylab.show()


