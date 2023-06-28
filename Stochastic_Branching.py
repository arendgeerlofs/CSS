import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit
from collections import Counter
from Subsampling import subsample

@jit
def branching_model(alpha, k, h, pdiss,size,iterations,subsample_sort=3):
    """
    Function that runs the Stochastic Branching model mapped on grid of neurons.
    The simulation is ran for all different parameter configurations given by the user.
    Main difference from BTW is that we consider random neighbors

    Input
    alpha: synaptic strength
    k: number of neighbors
    h: rate that changes the way that energy is conserved in the system
    pdiss: probability that a neuron is projected outside of the grid
    size: size of the grid (size x size)
    iterations: number of iterations of the simulation

    Output
    grid: grid from last frame
    grid_timed: grid from every frame
    avalanche_list: list of starting and end point of an avalanche
    aval_num: number of avalanches
    """
    avalanche_list = []
    spike_number = []

    # Initialize the grid of neurons
    grid= np.zeros((size,size))
    grid_new = np.zeros((size, size))
    grid_timed = []

    random_indices = np.random.choice(size * size, size=5, replace=False)
    grid.flat[random_indices] = 1

    activ_prob: float = alpha * (1 / k)
    subsample_matrix = subsample(grid, subsample_sort)
    for t in range(iterations):
        
        grid_new = grid.copy()
        grid_timed.append(grid_new)

        k_ij=[]
        for x in range(np.shape(grid)[0]):
                        for y in range(np.shape(grid)[1]):
                            if grid[x,y]==0:
                                k_ij.append([x,y])
        for i in range(np.shape(grid)[0]):
            for j in range(np.shape(grid)[1]):
                # If neuron spikes:
                if grid[i][j] == 1:
                    if np.random.random() < pdiss:
                                grid[i][j] = 0

                    kk = random.sample(k_ij,k)
                         
                            
                    for ij in kk:
                            if  np.random.poisson(h) < activ_prob:
                                grid[ij[0],ij[1]] = 1
                                grid[i][j] = 0
                                if subsample_matrix[i][j] == 1:

                                    avalanche_list.append([(i,j), (ij[0],ij[1]),t])
                                                  
                            else:
                                grid[ij[0],ij[1]] = 0
        num = np.sum(np.array(grid_timed[t-1]) > 0)
        spike_number.append(num)
    return grid, grid_timed, avalanche_list,spike_number

def count_avalanches_SB(data, search_length=100):
    '''
    This function counts the avalanches in every frame of the SB model.
    It does so by counting if there are neighboring neurons that spike.

    Input
    data: list of spikes that includes the beginning of the spike, the end and the frame it was on
    search_length: parameter for how many subsequent data points should be searched
    Output
    avalanche_count: list of unique avalanche identifiers.
    avalanche_durations: list of durations corresponding to each avalanche.
    is_in_aval: list indicating whether each frame is part of an avalanche.
    '''

    avalanche_count = 0
    avalanche_durations = []
    is_in_aval = [int(n) for n in range(len(data))]
    for t in range(len(data)):
      for i in range(search_length):
        if t+i < len(data) and data[t][1] == data[t+i][0] and data[t][2] == data[t+i][2] - 1:
           is_in_aval[t+i]=is_in_aval[t]

    avalanche_count = np.unique(is_in_aval)
    avalanche_durations=Counter(is_in_aval).values()
    return avalanche_count, avalanche_durations, is_in_aval



