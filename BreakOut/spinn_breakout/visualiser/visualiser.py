import enum
import numpy as np
import socket

import matplotlib.animation as animation
import matplotlib.colors as col
import matplotlib.pyplot as plt

BRIGHT_GREEN = (0.0, 0.9, 0.0)
# ----------------------------------------------------------------------------
# InputState
# ----------------------------------------------------------------------------
# Input states
class InputState(enum.IntEnum):
    idle = -1
    right = 0
    left = 1

# ----------------------------------------------------------------------------
# SpecialEvent
# ----------------------------------------------------------------------------
# Special events sent from game using first keys
class SpecialEvent(enum.IntEnum):
    score_up = 0
    score_down = 1
    max = 2

# ----------------------------------------------------------------------------
# Visualiser
# ----------------------------------------------------------------------------
class Visualiser(object):
    # How many bits are used to represent colour
    colour_bits = 1

    def __init__(self, udp_port, key_input_connection=None,
                 x_res=160, y_res=128, x_bits=8, y_bits=8):
        # Reset input state
        self.input_state = InputState.idle

        # Zero score
        self.score = 0

        # Cache reference to key input connection
        self.key_input_connection = key_input_connection

        # Build masks
        self.x_mask = (1 << x_bits) - 1
        self.x_shift = self.colour_bits + y_bits

        self.y_mask = (1 << y_bits) - 1
        self.y_shift = self.colour_bits

        self.colour_mask = (1 << self.colour_bits) - 1

        self.value_mask = (1 << (x_bits + y_bits + self.colour_bits)) - 1

        self.y_res=y_res
        self.x_res=x_res
        self.bat_width=32

        # Open socket to receive datagrams
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", udp_port))
        self.socket.setblocking(False)

        # Make awesome CRT palette
        cmap = col.ListedColormap(["black", BRIGHT_GREEN])

        # Create image plot to display game screen
        self.fig, self.axis = plt.subplots()
        self.image_data = np.zeros((y_res, x_res))
        self.image = self.axis.imshow(self.image_data, interpolation="nearest",
                                      cmap=cmap, vmin=0.0, vmax=1.0)

        # Draw score using textbox
        self.score_text = self.axis.text(0.5, 1.0, "0", color=BRIGHT_GREEN,
                                         transform=self.axis.transAxes,
                                         horizontalalignment="right",
                                         verticalalignment="top")
        # Hook key listeners
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        # Hide grid
        self.axis.grid(False)
        self.axis.set_xticklabels([])
        self.axis.set_yticklabels([])
        self.axis.axes.get_xaxis().set_visible(False)

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def show(self):
        # Play animation
        self.animation = animation.FuncAnimation(self.fig, self._update,
                                                 interval=20.0, blit=False)
        # Show animated plot (blocking)
        try:
            plt.show()
        except:
            pass

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------
    def _update(self, frame):
        # If state isn't idle, send spike to key input
        if self.input_state != InputState.idle and self.key_input_connection:
            self.key_input_connection.send_spike("key_input", self.input_state)

        # Read all datagrams received during last frame
        while True:
            try:
                raw_data = self.socket.recv(512)
            except socket.error:
                # If error isn't just a non-blocking read fail, print it
                # if e != "[Errno 11] Resource temporarily unavailable":
                #    print "Error '%s'" % e
                # Stop reading datagrams
                break
            else:
                # Slice off EIEIO header and convert to numpy array of uint32
                payload = np.fromstring(raw_data[6:], dtype="uint32")

                payload_value = payload & self.value_mask
                vision_event_mask = payload_value >= SpecialEvent.max
                # Payload is a pixel:
                # Create mask to select vision (rather than special event) packets

                # Extract coordinates
                vision_payload = payload_value[vision_event_mask] - SpecialEvent.max
                x = (vision_payload >> self.x_shift) & self.x_mask
                y = (vision_payload >> self.y_shift) & self.y_mask

                c = (vision_payload & self.colour_mask)


                '''if y.any() == self.y_res-1:
                    if c[np.where(y==self.y_res-1)].any()==1:
                        #add remaining bat pixels to image
                        x_pos=x[np.where(y==self.y_res-1)]
                        for i in range(1,self.bat_width):
                            np.hstack((y,self.y_res-1))
                            np.hstack((c,1))
                            np.hstack((x,x_pos+i))'''
                # Set valid pixels
                try:
                    self.image_data[y, x] = c
                except IndexError as e:
                    print("Packet contains invalid pixels:",
                          vision_payload, "X:", x, "  Y:",y, " c:",c)
                    self.image_data[:-1, :] = 0

                # Create masks to select score events and count them
                num_score_up_events = np.sum(payload_value == SpecialEvent.score_up)
                num_score_down_events = np.sum(payload_value == SpecialEvent.score_down)

                # If any score events occurred
                if num_score_up_events > 0 or num_score_down_events > 0:
                    # Apply to score count
                    self.score += num_score_up_events
                    self.score -= num_score_down_events

                    # Update displayed score count
                    self.score_text.set_text("%u" % self.score)

        # Set image data
        try:
            self.image.set_array(self.image_data)
        except NameError:
            pass

        # Return list of artists which we have updated
        # **YUCK** order of these dictates sort order
        # **YUCK** score_text must be returned whether it has
        # been updated or not to prevent overdraw
        return [self.image, self.score_text]

    def _on_key_press(self, event):
        # Send appropriate bits
        if event.key == "left":
            self.input_state = InputState.left
        elif event.key == "right":
            self.input_state = InputState.right

    def _on_key_release(self, event):
        # If either key is released set state to idle
        if event.key == "left" or event.key == "right":
            self.input_state = InputState.idle
