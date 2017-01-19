import numpy as np

# Magic constants -- NEED TO MANUALLY KEEP UPDATED TO MATCH C MODULE
GAME_WIDTH = 160
GAME_HEIGHT = 128
SPECIAL_EVENT_SCORE_UP = 0
SPECIAL_EVENT_SCORE_DOWN = 1
SPECIAL_EVENT_MAX = 2

COLOUR_ON = 1
COLOUR_OFF = 0

KEY_LEFT = 0
KEY_RIGHT = 1

on_neuron_ids = None
off_neuron_ids = None


def get_on_neuron_ids(width=GAME_WIDTH, height=GAME_HEIGHT):
    global on_neuron_ids
    if on_neuron_ids:
        return on_neuron_ids
    on_neuron_ids = np.zeros((height, width), dtype=np.int32)
    colour = COLOUR_ON
    for i in range(width):
        for j in range(height):
            on_neuron_ids[j, i] = SPECIAL_EVENT_MAX + (i << 9) + (j << 1) + colour
    return on_neuron_ids


def get_off_neuron_ids(width=GAME_WIDTH, height=GAME_HEIGHT):
    global off_neuron_ids
    if off_neuron_ids:
        return off_neuron_ids
    off_neuron_ids = np.zeros((height, width), dtype=np.int32)
    colour = COLOUR_OFF
    for i in range(width):
        for j in range(height):
            off_neuron_ids[j, i] = SPECIAL_EVENT_MAX + (i << 9) + (j << 1) + colour
    return off_neuron_ids


def get_reward_neuron_id():
    return SPECIAL_EVENT_SCORE_UP


def get_punishment_neuron_id():
    return SPECIAL_EVENT_SCORE_DOWN


def get_paddle_left_neuron_id():
    return KEY_LEFT


def get_paddle_right_neuron_id():
    return KEY_RIGHT


def create_pools(original_vector, pool_size):
    p = np.arange(original_vector.size)
    pooling_vector = p.reshape(p.size//pool_size, pool_size)
    return pooling_vector
