import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from collections import Counter

df = pd.read_csv("Data_neuron.txt", names=['spike times'], sep=" ")
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


bins = [2, 5, 10, 15, 20, 25, 50, 70, 110]
avalanches_per_bin = {}
branching_per_bin = {}
for binn in bins:
    avalanches_per_bin[binn], branching_per_bin[binn] = count_avalanches(bin_size=binn, time_line=time_line,
                                                                         in_vivo=in_vivo)

plt.figure(figsize=(11, 7))
s1 = []
for binn in bins:
    s1.append(Counter(avalanches_per_bin[binn])[1])

    frequency = Counter(avalanches_per_bin[binn])
    frequency = dict(sorted(frequency.items()))
    values, counts = zip(*frequency.items())
    # Plotting the line graph
    plt.loglog(values, counts, marker='o', linestyle='-', label=f'bin size={binn}')
    # Adding labels and title
    plt.xlabel('s, avalanche size')
    plt.ylabel('f(s), frequency in absolute counts')
    plt.title('Frequency plot of in vivo data')
    plt.legend(loc='best')
# Display the plot
plt.plot(values, 1 / np.asarray(values), linestyle='--', label=r'power-law $\tau=1$')
plt.legend()
plt.show()

plt.loglog(bins, s1, marker='o', linestyle='-')
plt.xlabel('Bin size(ms)')
plt.grid(True)
plt.ylabel('f(s=1, bs)')
plt.title('A different avalanche measure to assess criticality')
plt.show()

values_branch, counts_branch = zip(*branching_per_bin.items())
plt.figure(figsize=(11, 6))
plt.semilogx(values_branch, counts_branch, marker='o', linestyle='-', label=r'$\sigma^{*}$')
plt.axhline(1, color='black', linestyle='--')
plt.grid(True)
plt.legend()
plt.xlabel('Bin size(ms)')
plt.ylabel(r'$\sigma^{*}$, Branching parameter ')
plt.title(r'Estimated $\sigma^{*}$, changed with bin size')
plt.show()