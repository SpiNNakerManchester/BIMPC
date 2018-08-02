import spynnaker7.pyNN as p
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection

import pylab
import matplotlib.pyplot as plt
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
from spynnaker import plot_utils
import spinn_breakout
import threading
import time
from multiprocessing.pool import ThreadPool
import socket
import numpy as np

# Layout of pixels
from spynnaker.pyNN.models.utility_models.spike_injector import \
    SpikeInjector
from spinn_breakout.visualiser.visualiser import Visualiser

# def thread_score(udp_port):
#     score_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     socket.bind(("0.0.0.0", udp_port))
#     socket.setblocking(False)
#     try:
#         raw_data = socket.recv(512)
#     except socket.error:
#         # If error isn't just a non-blocking read fail, print it
#         # if e != "[Errno 11] Resource temporarily unavailable":
#         #    print "Error '%s'" % e
#         # Stop reading datagrams
#         break
#     else:
#         message_received = True
#         # Slice off EIEIO header and convert to numpy array of uint32
#         payload = np.fromstring(raw_data[6:], dtype="uint32")
#
#         payload_value = payload & self.value_mask
#         vision_event_mask = payload_value >= SpecialEvent.max
#         # Payload is a pixel:
#         # Create mask to select vision (rather than special event) packets
#
#         # Extract coordinates
#         vision_payload = payload_value[vision_event_mask] - SpecialEvent.max
#         x = (vision_payload >> self.x_shift) & self.x_mask
#         y = (vision_payload >> self.y_shift) & self.y_mask
#
#         c = (vision_payload & self.colour_mask)
#
#         '''if y.any() == self.y_res-1:
#             if c[np.where(y==self.y_res-1)].any()==1:
#                 #add remaining bat pixels to image
#                 x_pos=x[np.where(y==self.y_res-1)]
#                 for i in range(1,self.bat_width):
#                     np.hstack((y,self.y_res-1))
#                     np.hstack((c,1))
#                     np.hstack((x,x_pos+i))'''
#         # Set valid pixels
#         try:
#             # self.image_data[:] = 0
#             self.image_data[y, x] = c
#             # if c>0:
#             # self.video_data[:] = 0
#             # self.video_data[y, x, 1] = np.uint8(c*230)
#             # else:
#             #     self.video_data[y, x, :] = VIDEO_RED
#         except IndexError as e:
#             print("Packet contains invalid pixels:",
#                   vision_payload, "X:", x, "  Y:", y, " c:", c)
#             # self.image_data[:-1, :] = 0
#
#         # Create masks to select score events and count them
#         num_score_up_events = np.sum(payload_value == SpecialEvent.score_up)
#         num_score_down_events = np.sum(payload_value == SpecialEvent.score_down)
#
#         # If any score events occurred
#         if num_score_up_events > 0 or num_score_down_events > 0:
#             # Apply to score count
#             self.score += num_score_up_events
#             self.score -= num_score_down_events
#
#             # Update displayed score count
#             self.score_text.set_text("%u" % self.score)
#


def thread_visualiser(UDP_PORT):
    id = UDP_PORT - UDP_PORT1
    print "threadin ", running, id
    # time.sleep(5)
    visualiser = Visualiser(
        UDP_PORT, None, id,
        x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
        x_bits=X_BITS, y_bits=Y_BITS)
    print "threadin2 ", running, id
    # visualiser.show()
    visualiser._update(None)
    score = 0
    while running == True:
        print "in ", UDP_PORT, id, score
        score = visualiser._update(None)
        time.sleep(1)
    print "left ", running, id
    # score = visualiser._return_score()
    visual[id] = visualiser._return_image_data()
    result[id] = score

def row_col_to_input_breakout(row, col, is_on_input, row_bits, event_bits=1, colour_bits=1, row_start=0):
    row_bits = np.uint32(row_bits)
    idx = np.uint32(0)

    if is_on_input:
        idx = 1

    row += row_start
    idx = idx | (row << (colour_bits))  # colour bit
    idx = idx | (col << (row_bits + colour_bits))

    # add two to allow for special event bits
    idx = idx + 2

    return idx



def subsample_connection(x_res, y_res, subsamp_factor_x, subsamp_factor_y, weight,
                         coord_map_func):
    # subY_BITS=int(np.ceil(np.log2(y_res/subsamp_factor)))
    connection_list_on = []
    connection_list_off = []

    sx_res = int(x_res) // int(subsamp_factor_x)
    row_bits = 8#int(np.ceil(np.log2(y_res)))
    for j in range(int(y_res)):
        for i in range(int(x_res)):
            si = i // subsamp_factor_x
            sj = j // subsamp_factor_y
            # ON channels
            subsampidx = sj * sx_res + si
            connection_list_on.append((coord_map_func(j, i, 1, row_bits),
                                       subsampidx, weight, 1.))
            # OFF channels only on segment borders
            # if((j+1)%(y_res/subsamp_factor)==0 or (i+1)%(x_res/subsamp_factor)==0 or j==0 or i==0):
            connection_list_off.append((coord_map_func(j, i, 0, row_bits),
                                        subsampidx, weight, 1.))

    return connection_list_on, connection_list_off


X_BITS = 8
Y_BITS = 8

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT1 = 17887
UDP_PORT2 = UDP_PORT1 + 1

# Setup pyNN simulation
p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)

# Create breakout population and activate live output for it
breakout_pop = p.Population(1, spinn_breakout.Breakout, {}, label="breakout")
breakout_pop2 = p.Population(1, spinn_breakout.Breakout, {}, label="breakout")
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT1)
ex.activate_live_output_for(breakout_pop2, host="0.0.0.1", port=UDP_PORT2)


# Connect key spike injector to breakout population
rate = {'rate': 2}#, 'duration': 10000000}
spike_input = p.Population(2, p.SpikeSourcePoisson, rate, label="input_connect")
p.Projection(spike_input, breakout_pop, p.AllToAllConnector(weights=2))
spike_input2 = p.Population(2, p.SpikeSourcePoisson, rate, label="input_connect")
p.Projection(spike_input2, breakout_pop2, p.AllToAllConnector(weights=2))
# key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["input_connect"])

weight = 0.1
x_factor = 1
y_factor = 1
[Connections_on, Connections_off]=subsample_connection(X_RESOLUTION, Y_RESOLUTION, x_factor, y_factor, weight, row_col_to_input_breakout)
receive_pop_size = (160/x_factor)*(128/y_factor)
receive_pop_on = p.Population(receive_pop_size, p.IF_cond_exp, {}, label="receive_pop")
receive_pop_on.record()#["spikes"])
receive_pop_off = p.Population(receive_pop_size, p.IF_cond_exp, {}, label="receive_pop")
receive_pop_off.record()#["spikes"])
p.Projection(breakout_pop,receive_pop_on,p.FromListConnector(Connections_on))
p.Projection(breakout_pop,receive_pop_off,p.FromListConnector(Connections_off))

# Create visualiser
# visualiser = Visualiser(
#     UDP_PORT, None,
#     x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
#     x_bits=X_BITS, y_bits=Y_BITS)

running = True
t = threading.Thread(target=thread_visualiser, args=[UDP_PORT1])
r = threading.Thread(target=thread_visualiser, args=[UDP_PORT2])
result = [10 for i in range(2)]
x_res=160
y_res=128
visual = [np.zeros((y_res, x_res)) for i in range(2)]
# t = ThreadPool(processes=2)
# r = ThreadPool(processes=2)
# result = t.apply_async(thread_visualiser, [UDP_PORT1])
# result2 = r.apply_async(thread_visualiser, [UDP_PORT2])
# t.daemon = True
# Run simulation (non-blocking)
print "reached here 1"
t.start()
r.start()
runtime = 30000
p.run(runtime)
print "reached here 2"
running = False
# visualiser._return_score()

# Show visualiser (blocking)
# visualiser.show()

# spikes = []
# for j in range(receive_pop_size):
spikes_on = receive_pop_on.getSpikes()
pylab.figure()
ax = pylab.subplot(1, 2, 1)#4, 1)
pylab.plot([i[1] for i in spikes_on], [i[0] for i in spikes_on], "r.")
pylab.xlabel("Time (ms)")
pylab.ylabel("neuron ID")
pylab.axis([0, runtime, -1, receive_pop_size +1])

# ax = pylab.subplot(1, 4, 2)
# img = np.zeros((y_res//y_factor)*(x_res//x_factor))
# img[[i[0] for i in spikes_on]] = 1.
# plt.imshow(img.reshape((y_res//y_factor), (x_res//x_factor)), interpolation='none')

# pylab.show()

ax = pylab.subplot(1, 2, 2)#4, 3)
spikes_off = receive_pop_off.getSpikes()
pylab.plot([i[1] for i in spikes_off], [i[0] for i in spikes_off], "r.")
pylab.xlabel("Time (ms)")
pylab.ylabel("neuron ID")
pylab.axis([0, runtime, -1, receive_pop_size +1])


# ax = pylab.subplot(1, 4, 4)
# img = np.zeros((y_res//y_factor)*(x_res//x_factor))
# img[[i[0] for i in spikes_off]] = 1.
# plt.imshow(img.reshape((y_res//y_factor), (x_res//x_factor)), interpolation='none')


pylab.show()


# plt.figure(
#     # raster plot of the presynaptic neuron spike times
#     Panel(spikes[0].segments[0].spiketrains,
#           yticks=True, markersize=2, xlim=(0, 10)),
#     title="receive pop spikes",
#     annotations="Simulated with {}".format(p.name())
# )

# End simulation
p.end()
# score_result = result.get()
# score_result2 = result2.get()
while result[0] == 10 or result[1] == 10:
    None
print "result = ", result[0]
print "result2 = ", result[1]
print visual

while 1==1:
    None

