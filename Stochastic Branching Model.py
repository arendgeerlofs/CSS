import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit
from collections import Counter
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

@jit
def subsample(matrix, sort: float=1):
    if sort == 0:
        return np.ones(np.shape(matrix))
    sample = np.zeros(np.shape(matrix))
    size: int= 0
    dist: int = 0
    if sort <= 2:
        if sort == 1:
            size = 64
        else:
            size = 100
        i = 0
        while i < size:
            index = np.random.randint(np.shape(matrix)[0], size = 2)
            if sample[index[0]][index[1]] != 1:
                sample[index[0]][index[1]] = 1
                i += 1
    elif sort <= 6:
        if sort == 3:
            size = 64
            dist = 5
        elif sort == 4:
            size = 64
            dist = 1
        elif sort == 5:
            size = 16
            dist = 5
        else:
            size = 16
            dist = 1
        for i in range(int(np.sqrt(size))):
            for j in range(int(np.sqrt(size))):
                sample[4 + i*dist][4 + j*dist] = 1
    else:
        print("Invalid subsampling type, possible options are 1-6")
    return sample

def branching_model(alpha, k, h, pdiss,size,iterations):
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
    aval_num = []
    ID=0
    k_points = []
    spikes = []

    amount=0
    # Initialize the grid of neurons
    grid= np.zeros((size,size))
    grid_new = np.zeros((size, size))
    grid_ID = np.zeros((size, size))
    grid_timed = []

    # Initialising the grid with random 0 and 1
    random_indices = np.random.choice(size * size, size=1, replace=False)
    aaa =[]
    # Assign the value of 1 to the random positions
    grid.flat[random_indices] = 1

    activ_prob: float = alpha * (1 / k)
    # subsample_matrix = subsample(grid, subsample_sort)
    for t in range(iterations):
        if t % 1000==0:
            print(t)
        ID=1
        grid_new = grid.copy()
        grid_timed.append(grid_new)

        k_ij=[]
        for x in range(np.shape(grid)[0]):
                        for y in range(np.shape(grid)[1]):
                            if grid[x,y]==0:
                                k_ij.append([x,y])
        for i in range(np.shape(grid)[0]):
            for j in range(np.shape(grid)[1]):
                avalanche = 0 
                # If neuron spikes:
                if grid[i][j] == 1:
                    if np.random.random() < pdiss:
                                grid[i][j] = 0

                    # Activate each postsynaptic neuron with probability p
                    #n_i.append(random.sample(k_ij,k))
                    kk = random.sample(k_ij,k)

                    for ij in kk:
                            if  np.random.poisson(h) < activ_prob:
                                grid[ij[0],ij[1]] = 1
                                grid[i][j] = 0
                                k_points.append(k)
                                spikes.append(amount)
                                amount += 1
                                avalanche_list.append([(i,j), (ij[0],ij[1]),t])

                            else:
                                grid[ij[0],ij[1]] = 0
        if t>1:
            #avalanche_list.append(np.sum(np.array(grid_timed[t]-grid_timed[t-1]) > 0) )
            num = np.sum(np.array(grid_timed[t-1]) > 0)
            aval_num.append(np.sum(np.array(grid_timed[t]-grid_timed[t-1]) > 0)/num)
    return grid, grid_timed, avalanche_list,aval_num

# Set the parameters
#experiment by changing r

alpha = 0.99
k = 4
# r = [2.3,2.4,2.45,2.5,2.55,2.6]
r = 2.3
pdiss = 0.001

# Run the branching model
grid,u,aval,num = branching_model(alpha, k, r, pdiss, 50, 20000)


def plotheatmap(u_k, k):
    '''
    Heat map for the values of the model's grid.

    Input
    k: the current frame
    u_k: the grid at the current time frame k
    '''

    plt.clf()
    plt.xlabel("x ")
    plt.ylabel("y ")

    plt.pcolormesh(u_k, cmap='hot', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.ax.set_title('Spikes',fontsize=12)

    return plt

def animate(k):
    """
    Function that is used for the animation of the time evolution of the model.
    Input
    k: current frame
    """
    plt = plotheatmap(u[k], k)
    plt.title(f'Spikes at Frame {k}')

# frames = 1000
# anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=frames, repeat=False )
# anim.save("branching_heat.gif")

a = np.array(aval)

def count_avalanches_2(data):
    '''
    This function counts the avalanches in every frame of the SB model.
    It does so by counting if there are neighboring neurons that spike.

    Input
    data: list of spikes that includes the beginning of the spike, the end and the frame it was on

    Output
    avalanche_count: list of unique avalanche identifiers.
    avalanche_durations: list of durations corresponding to each avalanche.
    is_in_aval: list indicating whether each frame is part of an avalanche.

    '''

    avalanche_count = 0
    avalanche_durations = []
    is_in_aval = [int(n) for n in range(len(data))]
    for t in range(len(data)):
      for i in range(100):
        if t+i < len(data) and data[t][1] == data[t+i][0] and data[t][2] == data[t+i][2] - 1:
           is_in_aval[t+i]=is_in_aval[t]

    avalanche_count = np.unique(is_in_aval)
    avalanche_durations=Counter(is_in_aval).values()
    return avalanche_count, avalanche_durations, is_in_aval

avalanche_count, avalanche_durations_O, is_in_aval = count_avalanches_2(a)

#print("Number of avalanches:", avalanche_count)
#print("Avalanche durations:", avalanche_durations_O)
from scipy.optimize import curve_fit

plt.figure()
plt.plot(avalanche_durations_O)
plt.show()

plt.figure()
plt.hist(avalanche_durations_O, bins=30)
plt.show()
#sol2 = curve_fit(func_powerlaw, avalanche_count, avalanche_durations_O, p0 = np.asarray([-1,10**5,0]))
plt.figure()

bins_s = np.zeros(max(avalanche_durations_O)-1)
bins=[]
for i in range(1,max(avalanche_durations_O)):
    for j in range(len(avalanche_durations_O)):
        if i == list(avalanche_durations_O)[j]:
            bins_s[i-1] += 1
    bins.append(i)

plt.loglog(bins,bins_s)
plt.show()
    