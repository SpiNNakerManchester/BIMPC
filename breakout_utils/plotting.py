import matplotlib.pyplot as plt
import numpy as np


def plot_pools(original_vector, pooling_vector):
    fig = plt.figure()

    middle = pooling_vector.shape[1] // 2

    for x in np.ndenumerate(pooling_vector):
        y = pooling_vector[x[0][0], middle]
        plt.plot([x[1], y], [1, 0], color='b', alpha=.5)

    plt.scatter(original_vector, np.ones(original_vector.size))
    plt.scatter(pooling_vector[:, middle], np.zeros(pooling_vector.shape[0]))
    labels = ['Original Vector', 'Pooling vector']
    plt.yticks([1, 0], labels)
    plt.xticks([])
    plt.show()

if __name__ == "__main__":
    from breakout_utils.dealing_with_neuron_ids import create_pools, get_on_neuron_ids

    positions = np.arange(160)
    pool_size = 80

    pool = create_pools(positions, pool_size)
    plot_pools(positions, pool)

    ids = get_on_neuron_ids()

    first_row_pool = create_pools(ids[0, :], 80)
    plot_pools(np.arange(ids[0, :].size), first_row_pool)


