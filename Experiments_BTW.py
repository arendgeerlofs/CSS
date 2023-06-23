from Bak_Tang_Wiesenfield import BTW
from Heatmap import plotheatmap
import matplotlib.pyplot as plt
from itertools import groupby
from scipy import stats
import math
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from Subsampling import subsample
import numpy as np
from collections import Counter


# Temporal binning
def bins_count(bin_size, spikes):
    """
    Function that divides the time of the simulation into bins of different (given) lenghts

    Input:
    bin size: parameter that determines the size of the bins, that the timesteps will be discretised towards.
    spikes: this is a list that contains the number of spikes in the lattice per timestep.

    Output:
    aval_proper:
    """
    avalanches_pot = []
    # divide the time in bins
    for i in range(0, len(spikes), bin_size):
        group = spikes[i:i + bin_size]  # Slice the list to get the current group of elements
        group_sum = sum(group)  # Sum the elements in the current group
        avalanches_pot.append(group_sum)  # Append the group sum to the result list

    f = lambda x: x == 0
    aval_proper = [i for k, g in groupby(avalanches_pot, f) for i in (g if k else (sum(g),))]

    branch = []

    for i in range(0, len(avalanches_pot) - 1):
        group1 = avalanches_pot[i:i + 2]  # Slice the list to get the current group of elements
        if group1[1] != 0:
            group_division = group1[0] / group1[
                1]  # Divide the first element by the second element in the current group
            branch.append(group_division)
    branching_param = np.mean(branch)
    return aval_proper, branching_param


def animate(k, u):
    """
    Function that is used for the animation of the time evolution of the system/CA.
    Input: k, the current frame
    """
    plt = plotheatmap(u[k])

    plt.title(f'Heat distribution at Frame {k}')


# create the animation
# anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=frames, repeat=False)
# anim.save("spike_heat.gif")

def power_law_dist(configs, frames):
    for config in configs:
        u = BTW(50, 0, frames + 500, config[1], 'Param', config[0])

        # plot the spikes
        time = u[0][10:]
        spikes_num = u[1][10:]
        plt.plot(time, spikes_num)
        plt.xlabel('ms (Time frame)')
        plt.ylabel('number of spikes')
        plt.title(f'Avalanche sizes over time, alpha={config[0]}, h={config[1]}')
        plt.show()
        # temporal binning
        bins = [1, 2, 4, 8, 16, 32]
        avalanches_per_bin = {}
        for binn in bins:
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[0]

            frequency = Counter(avalanches_per_bin[binn])
            frequency = dict(sorted(frequency.items()))
            values, counts = zip(*frequency.items())

            # check if the plotted data actually follow a power law distribution
            fit_alpha, fit_loc, fit_scale = stats.powerlaw.fit(counts, loc=0)
            D, p = stats.kstest(counts, 'powerlaw', args=(fit_alpha, fit_loc, fit_scale))
            print(
                f"for alpha={config[0]}, h={config[1]} and {binn} bins, the KS test statistic is {D}")  # Kolmogorov-Smirnov test statistic
            print(f"for alpha={config[0]}, h={config[1]} and {binn} bins, the p-value is {p}")  # p-value of the test

            # Plotting the line graph
            plt.loglog(values, counts, linestyle='-', label=f'bin size={binn}')
            # Adding labels and title
            plt.xlabel('s, avalanche size')
            plt.xlim(1, None)
            plt.ylabel('f(s), frequency in absolute counts')
            plt.title(f'Avalanche size distribution f(s), alpha={config[0]}, h={config[1]}')
            plt.legend(loc='best')
        plt.show()


def branching_param_plot(configs, frames):
    store_data = {}
    plt.figure(figsize=(10,6))
    for config in configs:
        u = BTW(50, 0, frames + 500, config[1], 'Param', config[0])
        store_data[config] = u
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        branches_per_bin = {}
        for binn in bins:
            branches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[1]

        print(branches_per_bin.items())
        values_branch, counts_branch = zip(*branches_per_bin.items())
        plt.semilogx(values_branch, counts_branch, marker='o', linestyle='-', label=f' alpha={config[0]}, h={config[1]}')
        plt.axhline(1, color='black', linestyle='--')
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('Bin size(ms)')
        plt.ylabel(r'$\sigma$, Branching parameter ')
        plt.title(r'Estimated $\sigma$, changed with bin size')
    plt.show()

    return store_data


def new_avalanche_measures(configs, stored_data):
    for config in configs:
        u = stored_data[config]
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        avalanches_per_bin = {}
        for binn in bins:
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[0]

        s1 = []
        for binn in bins:
            s1.append(Counter(avalanches_per_bin[binn])[1])

        plt.loglog(bins, s1, marker='o', linestyle='-', label=f'alpha={config[0]}, h={config[1]}')
        plt.xlabel('Bin size(ms)')
        plt.grid(True)
        plt.ylabel('f(s=1, bs)')
        plt.legend(loc='best')
        plt.title(f'A different avalanche measure to assess criticality')
    plt.show()


power_law_configs = [(1, 5e-7), (1, 4e-5), (0.95, 0.0005), (0, 3)]
measures_branch_configs = [(0.9, 0.1), (0, 5), (0.99, 0.06), (0.999, 0.04), (1, 0.02), (0.98, 0.08), (1, 3e-7)]


# actual experiments and plots
power_law_dist(frames =49500, configs=power_law_configs)
stor_data = branching_param_plot(configs=measures_branch_configs, frames=49500)
new_avalanche_measures(configs=measures_branch_configs, stored_data=stor_data)
