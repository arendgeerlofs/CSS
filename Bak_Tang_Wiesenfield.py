import numpy as np
import matplotlib.pyplot as plt

def BTW(size, threshold, time_steps, labda, h, ret='BTW'):
    """
    """
    BTW = np.random.normal(0, 1, (size, size))
    BTW_new = np.zeros((size, size))
    BTW_spike_times = np.zeros((size, size))
    BTW_timed = []
    k_points = []
    spikes = []
    for k in range(time_steps):
        amount = 0
        for i in range(np.shape(BTW)[0]):
            for j in range(np.shape(BTW)[1]):
                if j + 1 < 50 and j - 1 >= 0 and i + 1 < 50 and i - 1 >= 0:
                    if BTW[i][j] > threshold:
                        BTW_new[i][j] = BTW[i][j] - 4
                        BTW_spike_times[i][j] = k
                        amount += 1
                    else:
                        BTW_new[i][j] = BTW[i][j] + 0.99 * (labda * (k - BTW_spike_times[i-1][j]) + labda * (k - BTW_spike_times[i+1][j])
                                    + labda * (k - BTW_spike_times[i][j-1]) + labda * (k - BTW_spike_times[i][j+1])) + np.random.poisson(h)
        BTW = BTW_new.copy()
        if ret = 'Param'
            if k % 10 == 0 and k > 1000:
                k_points.append(k)
                spikes.append(amount)
        if ret == 'Heat':
            BTW_timed.append(BTW)

    if ret == 'Heat':
        return BTW_timed
    if ret == 'Param':
        return k_points, spikes
    return BTW