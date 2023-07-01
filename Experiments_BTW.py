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
def bins_count(bin_size, spikes):
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
    # do not take into account the avalanches of size 0.
    f = lambda x: x == 0
    aval_proper = [i for k, g in groupby(avalanches_pot, f) for i in (g if k else (sum(g),))]

    branch = []
    # calculate the branching parameter
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

def power_law_dist(size, configs, frames, subsample_sort=0, refr=False, inh=False):
    """
    Function that runs the simulation of the CA, plots the number of spikes per frame
    and the distribution of avalanche sizes. The function also performs a statistical test,
    to determine whether the underlying distribution of avalanche sizes follows a power-law distribution.

    Input:
    configs: a tuple, that determines the configuration of "alpha" and "h" parameters of the simulation
    frames: the number of time frames (time steps) that will be performed in the simulation
    size: the size of the square grid. Will be used to demonstrate the robustness of criticality
    subsample_sort: the choice of subsampling strategy to be followed. In all our simulations, 100 neurons are randomly
    chosen and they are sampled out of the 50x50 grid.
    refr: boolean variable. When it is True the experiment assuming refractory period between the spiking of neurons is
    being performed.
    inh: boolean variable. When it is True the experiment assuming inhibitory connections between neurons is
    being performed.

    Output:
    The function produces the aforementioned plots and plots the results of the statistical tests
    """
    for config in configs:
        # run the actual simulation
        u = BTW(size, 0, frames + 500, config[1], 'Param', config[0], subsample_sort, refractory=refr, inhibitory=inh)

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
            # compute the avalanches per bin
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[0]
            # compute the f(s), frequency of avalanches for different sizes
            frequency = Counter(avalanches_per_bin[binn])
            frequency = dict(sorted(frequency.items()))
            values, counts = zip(*frequency.items())

            # check if the plotted data actually follow a power law distribution
            fit_alpha, fit_loc, fit_scale = stats.powerlaw.fit(counts, loc=0)
            D, p = stats.kstest(counts, 'powerlaw', args=(fit_alpha, fit_loc, fit_scale))
            if p > 0.05:
                exponent = stats.powerlaw.fit(counts)[0]
                print(f'The exponent of the power-law distribution is {exponent}')
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


def branching_param_plot(configs, frames, subsample_sort=0, refr=False, inh=False, mult_runs=False, configs_mult=None,
                         subsample_sort_mult=0):
    """
    Function that runs the simulation of the CA and plots the branching parameter per bin,
    for all the different parameter configurations given by the user.
    This experiment and the next one, have the same paramterer configurations. As a result of that, in order to save
    time and computational resources, this function stores the data from spikes and avalanches, to be used and plotted
    in the next experiment ( below function) without simulating of the system again.

    Input:
    configs: a tuple, that determines the configuration of "alpha" and "h" parameters of the simulation
    frames: the number of time frames (time steps) that will be performed in the simulation
    subsample_sort: the choice of subsampling strategy to be followed. In all our simulations, 100 neurons are randomly
    chosen and they are sampled out of the 50x50 grid.
    refr: boolean variable. When it is True the experiment assuming refractory period between the spiking of neurons is
    being performed.
    inh: boolean variable. When it is True the experiment assuming inhibitory connections between neurons is
    being performed.
    mult_runs: boolean variable. When it is True, we simulate the system multiple times and compute an estimate
    for the branching parameter.
    configs_mult: the parameter configurations for the above experiment
    subsample_sort_mult: the choice of subsampling strategy to be followed. In all our simulations, 100 neurons are randomly
    chosen and they are sampled out of the 50x50 grid. again this refers to the above experiment

    Output:
    store_data: This is a dictionary that stores the simulation data for each different parameter configuration.
    These data will be then used in the next experiment, to avoid simulating the system again.
    branches_per_bin: depending on the experiment, the function could also return a dictionary that contains the branching
    parameter per bin.
    """
    if not mult_runs:
        store_data = {}
        plt.figure(figsize=(10, 6))
        for config in configs:
            u = BTW(50, 0, frames + 500, config[1], 'Param', config[0], subsample_sort, refractory=refr, inhibitory=inh)
            store_data[config] = u
            spikes_num = u[1][10:]
            bins = [1, 2, 4, 8, 16, 32]
            branches_per_bin = {}
            for binn in bins:
                # again compute the branching parameter per bin and per parameter configuration
                branches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[1]
            # visualise the branching parameters
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

    if mult_runs:
        u = BTW(50, 0, frames + 500, configs_mult[1], 'Param', configs_mult[0],
                subsample_sort_mult, refractory=refr, inhibitory=inh)
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        branches_per_bin = {}
        for binn in bins:
            # again compute the branching parameter per bin and per parameter configuration
            branches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[1]

        _, counts_branch_mult = zip(*branches_per_bin.items())

        return counts_branch_mult


def new_avalanche_measures(configs, stored_data):
    """
    Function that plots 2 graphs that provide alternative avalanche measures.

    Input:
    store_data: This is a dictionary that stores the simulation data for each different parameter configuration,
    obtained from the previous experiment.
    configs: a tuple, that determines the configuration of "alpha" and "h" parameters of the simulation
    """
    for config in configs:
        # no need to run the simulation again, same parameter configurations with the above experiment
        u = stored_data[config]
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        avalanches_per_bin = {}
        for binn in bins:
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[0]
        # compute the f(s=1,bs) and visualise it
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

    # visualisation of the mean avalanche size per bin and per parameter configuration
    for config in configs:
        u = stored_data[config]
        spikes_num = u[1][10:]
        bins = [1, 2, 4, 8, 16, 32]
        avalanches_per_bin = {}
        mean_aval_size = []
        for binn in bins:
            avalanches_per_bin[binn] = bins_count(bin_size=binn, spikes=spikes_num)[0]
            mean_aval_size.append(np.mean(avalanches_per_bin[binn]))
        # plot of the mean avalanche size
        plt.loglog(bins, mean_aval_size, marker='o', linestyle='-', label=f'alpha={config[0]}, h={config[1]}')
        plt.xlabel('Bin size(ms)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title(r'Mean avalanche size for different values of $\alpha$')
        plt.ylabel('<s>, mean avalanche size')
    plt.show()


power_law_configs = [(1, 9.25e-8), (1, 1e-6), (0.95, 1e-4), (0, 8e-4)]
measures_branch_configs = [(1, 9.25e-8), (0.9, 0.0008), (0, 0.025), (0.99, 0.000002), (0.999, 0.000004), (1, 0.000001),
                           (0.98, 0.000009)]


# actual experiments and plots without subsampling
power_law_dist(size=50, frames=52500, configs=power_law_configs)
stor_data = branching_param_plot(configs=measures_branch_configs, frames=49500)
new_avalanche_measures(configs=measures_branch_configs, stored_data=stor_data)

grid_sizes = [40, 60]
for size_grid in grid_sizes:
    power_law_dist(size=size_grid, frames=99500, configs=power_law_configs)

# actual experiments and plots with subsampling
power_law_dist(size=50, frames=52500, configs=power_law_configs, subsample_sort=1)
stor_data_100 = branching_param_plot(configs=measures_branch_configs, frames=52500, subsample_sort=1)
new_avalanche_measures(configs=measures_branch_configs, stored_data=stor_data_100)

# experiments with the refractory period of neurons
power_law_dist(size=50, frames=14500, configs=power_law_configs, refr=True)
stor_data_refr = branching_param_plot(configs=measures_branch_configs, frames=14500, refr=True)
new_avalanche_measures(configs=measures_branch_configs, stored_data=stor_data_refr)

# experiments with inhibitory connections of neurons
power_law_dist(size=50, frames=14500, configs=power_law_configs, inh=True)
stor_data_inh = branching_param_plot(configs=measures_branch_configs, frames=14500, inh=True)
new_avalanche_measures(configs=measures_branch_configs, stored_data=stor_data_inh)


def branching_param_mult(subsample_choice):
    # Estimate the branching parameter for a specific parameter configration
    mult_branch_configs = [(0.9, 0.0008), (0.999, 4e-6)]
    bins = [1, 2, 4, 8, 16, 32]
    plt.figure()
    # for the different configurations of the parameters
    for param in mult_branch_configs:
        calc_dict = {}
        for i in range(15):
            # calculate the branching parameter, means, st.devs and confidence intervls
            calc_dict[i] = branching_param_plot(configs=param, frames=2500, mult_runs=True, configs_mult=param,
                                                subsample_sort_mult=subsample_choice)

        transposed_lists = np.transpose(list(calc_dict.values()))
        means = [np.mean(lst) for lst in transposed_lists]
        confidence_intervals = [stats.t.interval(0.95, len(lst) - 1, loc=np.mean(lst), scale=stats.sem(lst)) for lst in
                                transposed_lists]
        # plot the above calculated measures
        plt.semilogx(bins, means, 'o-', label=f'alpha={param[0]}, h={param[1]}')
        plt.fill_between(bins, [ci[0] for ci in confidence_intervals], [ci[1] for ci in confidence_intervals],
                         alpha=0.3, label='95% Confidence Interval')
        plt.xlabel('Bin size (ms)')
        plt.ylabel(r"$\sigma$, branching parameter")
        plt.title(r"Estimated $\sigma$, changed with bin size")
        plt.axhline(1, color='black', linestyle='--')
        plt.legend()
    plt.show()


subsample_sort = [0, 1]
for subsample_choice in subsample_sort:
    branching_param_mult(subsample_choice)
