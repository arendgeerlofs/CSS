import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from scipy import stats
from collections import Counter
from itertools import groupby

# load the in vivo data in the memory
df = pd.read_csv("Data_neuron.txt", names=['spike times'], sep=" ")
in_vivo = df['spike times'][:9500]

# plot the spikes of the neuron in time
plt.bar(in_vivo, np.ones(len(in_vivo)))
plt.xlabel('ms')
plt.ylabel('Spikes of neuron')
plt.show()

time_line = np.arange(0, math.ceil(in_vivo[len(in_vivo)-1]))


def count_avalanches(bin_size, time_line, in_vivo):
    """
    Function that counts the avalanches occurring in a bin, given the different bin size.
    The function also calculates the branching parameter for the given bin size.

    Input:
    bin_size: the desired bin size, given by the user. This determines how many timesteps are included in one bin.
    time_line: this a list that determines the time frame of the simulation.
    in_vivo: this is a list containing the spiking times of the 1 neuron in the in-vivo data.

    Output:
    aval_proper: this is a list that contains the number of avalanches occurring in each bin.
    branching: this is a list that contains the estimated branching parameter for every bin.
    """
    # determine the time interals created, according to the given bin size
    num_intervals = math.ceil(len(time_line) / bin_size)
    avalanches = []
    orig = 0
    pbar = tqdm(total=num_intervals, position=0, leave=True)
    count_prev = 0
    branching_params = []
    for i in range(num_intervals):
        # iterate through the bins
        inter = (orig, orig + bin_size)
        count = 0
        # count the number of spikes in the bins
        for j in range(len(in_vivo)):
            if (in_vivo[j] < inter[1] and in_vivo[j] > inter[0]):
                count += 1
        avalanches.append(count)
        # compute the branching parameter
        if count_prev != 0:
            branching_params.append(count / count_prev)
        count_prev = count

        orig += bin_size
        pbar.update()
    pbar.close()
    branching = np.mean(branching_params)
    # remove the zero avalanches and compute the avalanches as the consecutive nonzero bins.
    f = lambda x: x == 0
    aval_proper = [i for k, g in groupby(avalanches, f) for i in (g if k else (sum(g),))]

    return aval_proper, branching


bins = [1, 2, 4, 8, 16, 32, 64]
avalanches_per_bin = {}
branching_per_bin = {}
for binn in bins:
    # compute the avalanches per bin
    avalanches_per_bin[binn], branching_per_bin[binn] = count_avalanches(bin_size=binn, time_line=time_line,
                                                                         in_vivo=in_vivo)

plt.figure(figsize=(11, 7))
s1 = []
for binn in bins:
    # compute the frequency of single events f(s=1)
    s1.append(Counter(avalanches_per_bin[binn])[1])

    frequency = Counter(avalanches_per_bin[binn])
    frequency = dict(sorted(frequency.items()))
    values, counts = zip(*frequency.items())
    # estimate the exponent of power-law(if any) and run statistical tests
    # for power law distributions of the generated data
    fit_alpha, fit_loc, fit_scale = stats.powerlaw.fit(counts, loc=0)
    D, p = stats.kstest(counts, 'powerlaw', args=(fit_alpha, fit_loc, fit_scale))
    if p > 0.05:
        exponent = stats.powerlaw.fit(counts)[0]
        print(f'The exponent of the power-law distribution is {exponent}')
    print(f"for {binn} bins, the KS test statistic is {D}")  # Kolmogorov-Smirnov test statistic
    print(f"for {binn} bins, the p-value is {p}")  # p-value of the test

    # Plotting the line graph
    plt.loglog(values, counts, marker='o', linestyle='-', label=f'bin size={binn}')
    # Adding labels and title
    plt.xlabel('s, avalanche size')
    plt.ylabel('f(s), frequency in absolute counts')
    plt.title('Frequency plot of in vivo data')
    plt.legend(loc='best')
# Display the plot
plt.legend()
plt.show()

# visualise the frequency of avalanches of size 1
plt.loglog(bins, s1, marker='o', linestyle='-')
plt.xlabel('Bin size(ms)')
plt.grid(True)
plt.ylabel('f(s=1, bs)')
plt.title('A different avalanche measure to assess criticality')
plt.show()

# visualise the branching parameter
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