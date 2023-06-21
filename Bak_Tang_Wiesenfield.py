import numpy as np
from numba import jit

@jit
def BTW(size = 50, threshold = 0, time_steps = 1000, labda = 0.05, h = 0, ret='BTW'):
    """
    """
    alpha = 0.99
    BTW = np.random.normal(-1.27, 1, (size, size))
    BTW_new = np.zeros((size, size))
    BTW_spike_times = np.zeros((size, size))
    BTW_timed = [BTW]
    k_points = []
    spikes = []
    # update the whole matrix for each time step and keep track of the amount of spike per timestep
    for k in range(time_steps):
        amount = 0
        for i in range(np.shape(BTW)[0]):
            for j in range(np.shape(BTW)[1]):
                if BTW[i][j] > threshold:
                    # update function if above threshold
                    BTW_new[i][j] = BTW[i][j] - 4
                    BTW_spike_times[i][j] = k
                    amount += 1
                else:
                    BTW_new[i][j] = BTW[i][j] + np.random.poisson(h)
                    if j + 1 < 50:
                        BTW_new[i][j] += alpha/1000 * (labda * (k - BTW_spike_times[i][j+1]))
                        if BTW_spike_times[i][j+1] == k - 1:
                            BTW_new[i][j] += 1
                    if j - 1 >= 0:
                        BTW_new[i][j] += alpha/1000 * (labda * (k - BTW_spike_times[i][j-1]))
                        if BTW_spike_times[i][j-1] == k - 1:
                            BTW_new[i][j] += 1
                    if i + 1 < 50:
                        BTW_new[i][j] += alpha/1000 * (labda * (k - BTW_spike_times[i+1][j]))
                        if BTW_spike_times[i+1][j] == k - 1:
                            BTW_new[i][j] += 1
                    if i - 1 >= 0:
                        BTW_new[i][j] += alpha/1000 * (labda * (k - BTW_spike_times[i-1][j]))
                        if BTW_spike_times[i-1][j] == k - 1:
                            BTW_new[i][j] += 1
        BTW = BTW_new.copy()
        if ret == 'Param':
            if k % 10 == 0:
                k_points.append(k)
                spikes.append(amount)
                print(k, amount)
        if ret == 'Heat':
            BTW_timed.append(BTW)
    if ret == 'Heat':
        return BTW_timed
    if ret == 'Param':
        return k_points, spikes
    return BTW

BTW(50, 0, 500, 0.1, 0, ret='Param')
