import spynnaker7.pyNN as p
# from spynnaker_external_devices_plugin.pyNN.connections.\
#     spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.connections.\
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
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
X_BITS = 8

def thread_visualiser(UDP_PORT):
    print "threadin ", running
    # time.sleep(5)
    visualiser = Visualiser(
        UDP_PORT, None,
        x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
        x_bits=X_BITS, y_bits=Y_BITS)
    visualiser.show()
    # visualiser._update()
    while running == True:
        print "in ", UDP_PORT
        # visualiser._update()
        time.sleep(1)
    print "left ", running
    score = visualiser._return_score()
    return score


Y_BITS = 8

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT1 = 17893
UDP_PORT2 = 17894

# Setup pyNN simulation
p.setup(timestep=1.0)

# Create breakout population and activate live output for it
breakout_pop = p.Population(1, spinn_breakout.Breakout, {}, label="breakout")
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT1)
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT2)

# Create spike injector to send random spikes to the paddle
spike_array = [
    [0, 2, 4, 6, 8, 10],
    [1, 3, 5, 7, 9, 11]
    ]

# Connect key spike injector to breakout population
# array_input = p.Population(2, p.SpikeSourceArray(spike_times=spike_array), label="input_connect")
# poisson = p.SpikeSourcePoisson(rate=20)
rate = {'rate': 2}#, 'duration': 10000000}
spike_input = p.Population(2, p.SpikeSourcePoisson, rate, label="input_connect")
p.Projection(spike_input, breakout_pop, p.AllToAllConnector(weights=2))
# key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["input_connect"])

# Create visualiser
# visualiser = Visualiser(
#     UDP_PORT, None,
#     x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
#     x_bits=X_BITS, y_bits=Y_BITS)

running = True
# t = threading.Thread(target=thread_visualiser, args=())
t = ThreadPool(processes=2)
r = ThreadPool(processes=2)
result = t.apply_async(thread_visualiser, [UDP_PORT1])
result2 = r.apply_async(thread_visualiser, [UDP_PORT2])
# t.daemon = True
# Run simulation (non-blocking)
print "reached here 1"
# t.start()
p.run(10000)
print "reached here 2"
# visualiser._return_score()

# Show visualiser (blocking)
# visualiser.show()


# End simulation
p.end()
running = False
score_result = result.get()
score_result2 = result2.get()
print "result = ", score_result
print "result2 = ", score_result2

while 1==1:
    None

