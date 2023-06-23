import numpy as np
from numba import jit
from tqdm import tqdm
from scipy.stats import poisson


def generate_poisson(rate, cell_val):
    a=poisson.rvs(rate)
    if a > 0:
        x = -cell_val + 0.01
    else:
        x = 0
    #print(f'number is {a}')
    return x


def BTW(size=50, threshold=0, time_steps=1000, h=0.91, ret='BTW', alpha=1):
    """
    """
    BTW = np.random.normal(-2, 1, (size, size))
    BTW_new = np.zeros((size, size))
    BTW_spike_times = np.full((size, size), -1)
    BTW_spike_times_new = np.full((size, size), -1)
    BTW_heatmap = np.zeros((50, 50))
    BTW_timed = []
    k_points = []
    spikes = []
    last_amount = 0
    # update the whole matrix for each time step and keep track of the amount of spike per timestep
    pbar = tqdm(total=time_steps, position=0, leave=True)
    for k in range(1, time_steps):
        amount = 0
        for i in range(np.shape(BTW)[0]):
            for j in range(np.shape(BTW)[1]):
                poi = generate_poisson(h, BTW[i][j])
                BTW_new[i][j] = BTW[i][j] + poi
                if j + 1 < 50:
                    if k - BTW_spike_times[i][j + 1] == 1:
                        BTW_new[i][j] += alpha
                if j - 1 >= 0:
                    if k - BTW_spike_times[i][j - 1] == 1:
                        BTW_new[i][j] += alpha
                if i + 1 < 50:
                    if k - BTW_spike_times[i + 1][j] == 1:
                        BTW_new[i][j] += alpha
                if i - 1 >= 0:
                    if k - BTW_spike_times[i - 1][j] == 1:
                        BTW_new[i][j] += alpha
                if BTW_new[i][j] > threshold:
                    BTW_heatmap[i][j] = 1
                    # update function if above threshold
                    BTW_new[i][j] = BTW_new[i][j] - 4
                    BTW_spike_times_new[i][j] = k
                    amount += 1
                else:
                    BTW_heatmap[i][j] = 0
        BTW = BTW_new.copy()
        BTW_spike_times = BTW_spike_times_new.copy()
        last_amount = amount
        if ret == 'Param':
            if True:
                k_points.append(k)
                spikes.append(amount)
                #print(k, amount)
        if ret == 'Heat' and k >= 500:
            BTW_timed.append(BTW_heatmap)
        pbar.update()
    pbar.close()
    if ret == 'Heat':
        return BTW_timed
    if ret == 'Param':
        return k_points, spikes
    return BTW
