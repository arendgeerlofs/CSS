import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from collections import Counter

df = pd.read_csv("Desktop/Data_neuron.txt", names=['spike times'], sep=" ")
in_vivo = df['spike times'][:9500]

plt.bar(in_vivo, np.ones(len(in_vivo)))
plt.xlabel('ms')
plt.ylabel('Spikes of neuron')
plt.show()

time_line = np.arange(0, math.ceil(in_vivo[len(in_vivo)-1]))


def count_avalanches(bin_size, time_line, in_vivo):
    num_intervals = math.ceil(len(time_line) / bin_size)
    avalanches = []
    orig = 0
    pbar = tqdm(total=num_intervals, position=0, leave=True)
    count_prev = 0
    branching_params = []
    for i in range(num_intervals):
        inter = (orig, orig + bin_size)
        count = 0
        for j in range(len(in_vivo)):
            if (in_vivo[j] < inter[1] and in_vivo[j] > inter[0]):
                count += 1
        avalanches.append(count)

        if count_prev != 0:
            branching_params.append(count / count_prev)
        count_prev = count

        orig += bin_size
        pbar.update()
    pbar.close()
    branching = np.mean(branching_params)

    return avalanches, branching