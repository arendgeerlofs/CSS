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


def subsampling_random(spikes_per_bin):
    """
    Function that performs random subsampling in the spikes per bin, for a simulation.
    The function removes randomly some spikes in the non-zero bins.
    By doing that, the random subsampling is implemented.

    Input:
    spikes_per_bin: list that contains the number of spikes per bin

    Output:
    spikes_per_bin: modified list, that also accounts for subsampling, in the way that was described above.
    """
    for i in range(len(spikes_per_bin)):
        if spikes_per_bin[i] != 0:
            spikes_per_bin[i] -= int(np.random.randint(low=0, high=spikes_per_bin[i]+1, size=1))

    return spikes_per_bin


def remove_zero_vals(tple):
    """
    Function that removes the avalanches of size zero (no avalanches) from a given tuple.
    This will be used for the visualisation of avalanche size distribution

    Input:
    tple: tuple that contains all the different avalanche sizes per bin

    Output:
    tuple_ret: the updated tuple, with all the zero values removed
    """
    # Convert the tuple to a list
    lst = list(tple)

    # Remove all zero values from the list
    lst = [x for x in lst if x != 0]

    # Convert the list back to a tuple
    tuple_ret = tuple(lst)
    return tuple_ret


# Temporal binning
def bins_count(bin_size, spikes, subsampling=False):
    """
    Function that divides the time of the simulation into bins of different (given) lenghts

    Input:
    bin size: parameter that determines the size of the bins, that the timesteps will be discretised towards.
    spikes: this is a list that contains the number of spikes in the lattice per timestep.

    Output:
    aval_proper: this is a list that contains the number of avalanches per bin, for the desired bin size
    branching_param: this is the estimated branching parameter for the desired bin size
    """
    avalanches_pot = []
    # divide the time in bins
    for i in range(0, len(spikes), bin_size):
        group = spikes[i:i + bin_size]  # Slice the list to get the current group of elements
        group_sum = sum(group)  # Sum the elements in the current group
        avalanches_pot.append(group_sum)  # Append the group sum to the result list
    if subsampling:
        avalanches_pot = subsampling_random(avalanches_pot)
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
    Input:
    k: this is the current frame
    u: this is the number of spikes in the lattice per timestep
    """
    plt = plotheatmap(u[k])

    plt.title(f'Heat distribution at Frame {k}')


# create the animation
# anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=frames, repeat=False)
# anim.save("spike_heat.gif")

def power_law_dist(size, configs, frames, subsample_sort=0):
    """
    Function that runs the simulation of the CA, plots the number of spikes per frame
    and the distribution of avalanche sizes. The function also performs a statistical test,
    to determine whether the underlying distribution of avalanche sizes follows a power-law distribution.

    Input:
    configs: a tuple, that determines the configuration of "alpha" and "h" parameters of the simulation
    frames: the number of time frames (time steps) that will be performed in the simulation
    size: the size of the square grid. Will be used to demonstrate the robustness of criticality

    Output:
    The function produces the aforementioned plots and plots the results of the statistical tests
    """
    for config in configs:
        u = BTW(size, 0, frames + 500, config[1], 'Param', config[0], subsample_sort)

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
        subsample_p = True if config[1] > 1e-8 else False
        for binn in bins:
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num, subsampling=subsample_p)[0]

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
            plt.loglog(values, remove_zero_vals(counts), linestyle='-', label=f'bin size={binn}')
            # Adding labels and title
            plt.xlabel('s, avalanche size')
            plt.ylabel('f(s), frequency in absolute counts')
            plt.title(f'Avalanche size distribution f(s), alpha={config[0]}, h={config[1]}')
            plt.legend(loc='best')
        plt.show()


def branching_param_plot(configs, frames, subsample_sort=0):
    """
    Function that runs the simulation of the CA and plots the branching parameter per bin,
    for all the different parameter configurations given by the user.
    This experiment and the next one, have the same paramterer configurations. As a result of that, in order to save
    time and computational resources, this function stores the data from spikes and avalanches, to be used and plotted
    in the next experiment ( below function) without simulating of the system again.

    Input:
    configs: a tuple, that determines the configuration of "alpha" and "h" parameters of the simulation
    frames: the number of time frames (time steps) that will be performed in the simulation

    Output:
    store_data: This is a dictionary that stores the simulation data for each different parameter configuration.
    These data will be then used in the next experiment, to avoid simulating the system again.
    """
    store_data = {}
    plt.figure(figsize=(10, 6))
    for config in configs:
        u = BTW(50, 0, frames + 500, config[1], 'Param', config[0], subsample_sort)
        store_data[config] = u
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        branches_per_bin = {}
        subsample_p = True if config[1] > 1e-8 else False
        for binn in bins:
            branches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num, subsampling=subsample_p)[1]

        print(branches_per_bin.items())
        values_branch, counts_branch = zip(*branches_per_bin.items())
        plt.semilogx(values_branch, counts_branch, marker='o', linestyle='-',
                     label=f' alpha={config[0]}, h={config[1]}')
        plt.axhline(1, color='black', linestyle='--')
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('Bin size(ms)')
        plt.ylabel(r'$\sigma$, Branching parameter ')
        plt.title(r'Estimated $\sigma$, changed with bin size')
    plt.show()

    return store_data


def new_avalanche_measures(configs, stored_data):
    """
    Function that plots 2 graphs that provide alternative avalanche measures.

    Input:
    store_data: This is a dictionary that stores the simulation data for each different parameter configuration,
    obtained from the previous experiment.
    configs: a tuple, that determines the configuration of "alpha" and "h" parameters of the simulation
    """
    for config in configs:
        u = stored_data[config]
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        avalanches_per_bin = {}
        for binn in bins:
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num, subsampling=False)[0]

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

    for config in configs:
        u = stored_data[config]
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        avalanches_per_bin = {}
        mean_aval_size = []
        for binn in bins:
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[0]
            mean_aval_size.append(np.mean(avalanches_per_bin[binn]))

        plt.loglog(bins, mean_aval_size, marker='o', linestyle='-', label=f'alpha={config[0]}, h={config[1]}')
        plt.xlabel('Bin size(ms)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title(r'Mean avalanche size for different values of $\alpha$')
        plt.ylabel('<s>, mean avalanche size')
    plt.show()


# (1, 9.25e-8),
power_law_configs = [(1, 1e-6), (0.95, 1e-4), (0, 8e-4)]
measures_branch_configs = [(0.9, 0.0008), (0, 0.025), (0.99, 0.00002), (0.999, 0.00004), (1, 0.00001), (0.98, 0.00006),
                           (1, 9.25e-8)]

# actual experiments and plots without subsampling
power_law_dist(size=50, frames=54500, configs=power_law_configs)
stor_data = branching_param_plot(configs=measures_branch_configs, frames=54500)
new_avalanche_measures(configs=measures_branch_configs, stored_data=stor_data)


grid_sizes = [25, 35, 40]

for size_grid in grid_sizes:
    power_law_dist(size=size_grid, configs=power_law_configs[0], frames=54500)

# actual experiments and plots
power_law_dist_100(size=50, frames=54500, configs=power_law_configs, subsample_sort = 1)
stor_data_100 = branching_param_plot(configs=measures_branch_configs, frames=54500, subsample_sort = 1)
new_avalanche_measures_100(configs=measures_branch_configs, stored_data=stor_data_100)
