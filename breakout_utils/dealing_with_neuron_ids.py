import numpy as np
from constants import *


def get_on_neuron_ids(width=GAME_WIDTH, height=GAME_HEIGHT):
    on_neuron_ids = np.zeros((height, width), dtype=np.int32)
    colour = COLOUR_ON
    for i in range(width):
        for j in range(height):
            on_neuron_ids[j, i] = SPECIAL_EVENT_MAX + (i << 9) + (j << 1) + colour
    return on_neuron_ids


def get_off_neuron_ids(width=GAME_WIDTH, height=GAME_HEIGHT):
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
