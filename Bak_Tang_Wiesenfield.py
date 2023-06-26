import numpy as np
from numba import jit
from tqdm import tqdm
from scipy.stats import poisson
from Subsampling import subsample


def generate_poisson(rate, cell_val):
    a=poisson.rvs(rate)
    if a > 0:
        x = -cell_val + 0.01
    else:
        x = 0
    #print(f'number is {a}')
    return x


def BTW(size=50, threshold=0, time_steps=1000, h=0.91, ret='BTW', alpha=1, subsample_sort=0):
    """

    """
    # initialize grid for k and k+1
    BTW = np.random.normal(-2, 1, (size, size))
    BTW_new = np.zeros((size, size))

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
        BTW_heatmap = np.zeros((50, 50))
        amount = 0
        for i in range(np.shape(BTW)[0]):
            for j in range(np.shape(BTW)[1]):
                # update value
                poi = generate_poisson(h, BTW[i][j])
                BTW_new[i][j] = BTW[i][j] + poi
                if j + 1 < size:
                    if k - BTW_spike_times_last[i][j + 1] == 1:
                        BTW_new[i][j] += alpha
                if j - 1 >= 0:
                    if k - BTW_spike_times_last[i][j - 1] == 1:
                        BTW_new[i][j] += alpha
                if i + 1 < size:
                    if k - BTW_spike_times_last[i + 1][j] == 1:
                        BTW_new[i][j] += alpha
                if i - 1 >= 0:
                    if k - BTW_spike_times_last[i - 1][j] == 1:
                        BTW_new[i][j] += alpha
                
                # update again if it activates
                if BTW_new[i][j] > threshold:
                    BTW_heatmap[i][j] = 1
                    BTW_new[i][j] = BTW_new[i][j] - 4
                    BTW_spike_times[i][j] = k
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
