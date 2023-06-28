import numpy as np
from numba import jit
from tqdm import tqdm
from scipy.stats import poisson
from Subsampling import subsample

@jit
def generate_poisson(rate, cell_val):
    """
    Function that makes neuron spike with certain poisson rate

    Input
    rate: poisson rate
    cell_val: current value of neuron

    Output
    update_value: change in cell value
    """
    p = poisson.rvs(rate)
    # activate neuron
    if p > 0:
        update_value = -cell_val + 0.01
    # no change
    else:
        update_value = 0
    return update_value

def direct_neighbor(cell, neighbor):
    """
    Function that checks if a grid point is directly neighboring another grid point

    Input
    cell: list of x and y index indicating grid point
    neighbor: list of x and y index indicating grid point

    Output
    binary value indicating if the two grid points are direct neighbors or not
    """
    # check for horizontal neighbors
    if cell[0] == neighbor[0]:
        if cell[1] - neighbor[1] == -1 or cell[1] - neighbor[1] == 1:
            return True
    # check for vertical neighbors
    if cell[1] == neighbor[1]:
        if cell[0] - neighbor[0] == -1 or cell[0] - neighbor[0] == 1:
            return True
    return False

def inhibitory_neighbors(BTW_shape, alpha):
    """
    Function that add a random amount of random neighbors and alters synaptic strength based on amount of total neighbors

    Input
    BTW_shape: shape of original BTW grid
    alpha: synaptic strength for each neuron

    Output
    neighbor_dict: dictionary with all added neighbors for each neuron
    alpha: updated synaptic strength for each neuron
    """
    neighbor_dict = {}
    for i in range(BTW_shape[0]):
        for j in range(BTW_shape[1]):
            neighbor_list = []
            # compute amount of extra neighbors
            extra_neighbors = np.random.poisson(2)
            for k in range(extra_neighbors):
                added = False
                # only add grid point to neighbor list if not already a neighbor or itself
                while not added:
                    index_i = np.random.randint(BTW_shape[0])
                    index_j = np.random.randint(BTW_shape[1])
                    if not (index_i == i and index_j == j) and not direct_neighbor([i, j], [index_i, index_j]):
                        if [index_i, index_j] not in neighbor_list:
                            neighbor_list.append([index_i, index_j])
                            added = True

            # recalculate synaptic strength by normalizing on new amount of total neighbors
            alpha[i][j] = (alpha[i][j]*4) / (4 + extra_neighbors)
            neighbor_dict[str(i) + " - " + str(j)] = neighbor_list
    return neighbor_dict, alpha


def BTW(size=50, threshold=0, time_steps=1000, h=0.91, ret='BTW', alpha=1, subsample_sort=0, refractory = False, inhibitory = False):
    """
    Function that runs the Bak Tang Wiesenfield model mapped on a grid of neurons.

    Input
    size: size of square grid
    threshold: value at which above it a neuron spikes
    time_steps: amount of time steps for which the model runs
    h: poisson rate for randomly making a neuron spike
    ret: type of return funcion
    alpha: synaptic strength
    subsample_sort: type of subsampling applied
    refractory: binary if neuron value can change in next 10 time steps after spiking
    inhibitory: binary if neurons have extra random neighbors of longer synaptic length

    Output
    BTW_heatmap_list: list with grid indicating which neurons spike at each time steps
    k_points: list of k values indicating time steps
    spikes: list of amounts of spikes at each time step
    BTW: resulting BTW grid with values of the neurons
    """
    # initialize grid for k and k+1
    BTW = np.random.normal(-2, 1, (size, size))
    BTW_new = np.zeros((size, size))

    # set synaptic strength for each grid point
    alpha = np.full((size, size), alpha)

    # initialize extra neighbors
    if inhibitory:
        neighbor_dict, alpha = inhibitory_neighbors(np.shape(BTW), alpha)

    # initialize neuron spike for k and k-1
    BTW_spike_times_last = np.full((size, size), -1)
    BTW_spike_times = np.full((size, size), -1)

    # neuron activation grid 
    BTW_heatmap_list = []

    # get subsample
    subsample_matrix = subsample(BTW, subsample_sort)

    # list of k values and spikes for visualization per timestep
    k_points = []
    spikes = []

    # update the whole matrix for each time step and keep track of the amount of spikes per timestep
    pbar = tqdm(total=time_steps, position=0, leave=True)
    for k in range(1, time_steps):
        BTW_heatmap = np.zeros((size, size))
        amount = 0
        for i in range(np.shape(BTW)[0]):
            for j in range(np.shape(BTW)[1]):
                # update value based on direct neighbors
                if k - BTW_spike_times[i][j] > 10 or k < 10 or not refractory:
                    poi = generate_poisson(h, BTW[i][j])
                    BTW_new[i][j] = BTW[i][j] + poi
                    if j + 1 < size:
                        if k - BTW_spike_times_last[i][j + 1] == 1:
                            BTW_new[i][j] += alpha[i][j]
                    if j - 1 >= 0:
                        if k - BTW_spike_times_last[i][j - 1] == 1:
                            BTW_new[i][j] += alpha[i][j]
                    if i + 1 < size:
                        if k - BTW_spike_times_last[i + 1][j] == 1:
                            BTW_new[i][j] += alpha[i][j]
                    if i - 1 >= 0:
                        if k - BTW_spike_times_last[i - 1][j] == 1:
                            BTW_new[i][j] += alpha[i][j]
                
                # update based on inhibitory neighbors
                if inhibitory:
                    for neighbor in neighbor_dict[str(i) + " - " + str(j)]:
                        if k - BTW_spike_times_last[neighbor[0]][neighbor[1]] == 1:
                            BTW_new[i][j] += alpha[i][j]
                    
                # update again if it activates
                if BTW_new[i][j] > threshold:
                    BTW_heatmap[i][j] = 1
                    BTW_new[i][j] = BTW_new[i][j] - 4
                    BTW_spike_times[i][j] = k
                    # only count spike if neuron in subsample
                    if subsample_matrix[i][j] == 1:
                        amount += 1

        BTW = BTW_new.copy()
        BTW_spike_times_last = BTW_spike_times.copy()

        if ret == 'Param':
            k_points.append(k)
            spikes.append(amount)
        if ret == 'Heat' and k >= 500:
            BTW_heatmap_list.append(BTW_heatmap)
        pbar.update()

    pbar.close()

    if ret == 'Heat':
        return BTW_heatmap_list
    if ret == 'Param':
        return k_points, spikes
    return BTW