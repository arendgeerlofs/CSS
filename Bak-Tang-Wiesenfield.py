import numpy as np

def BTW(size, threshold, time_steps, labda, h):
    """
    """
    BTW = np.random.normal(0, 1, (size, size))
    BTW_new = np.zeros((size, size))
    BTW_spike_times = np.zeros((size, size))
    for k in range(time_steps):
        print("------------------")
        for i in range(np.shape(BTW)[0]):
            for j in range(np.shape(BTW)[1]):
                if j + 1 < 50 and j - 1 >= 0 and i + 1 < 50 and i - 1 >= 0:
                    if BTW[i][j] > threshold:
                        BTW_new[i][j] = BTW[i][j] - 4
                        BTW_spike_times[i][j] = k
                        print((i, j), k)
                    else:
                        BTW_new[i][j] = BTW[i][j] + 0.25 * (labda * (k - BTW[i-1][j]) + labda * (k - BTW[i+1][j])
                                    + labda * (k - BTW[i][j-1]) + labda * (k - BTW[i][j+1])) + np.random.poisson(h)
        BTW = BTW_new.copy()
    return BTW

BTW(50, 0, 100, 0.005, 0.5)
