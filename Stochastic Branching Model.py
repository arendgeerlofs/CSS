import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
@jit
def choose_random_values(grid, num_values):
    
    # flatten the grid into a 1D array
    flattened_grid = grid.flatten()

    # Randomly choose num_values elements from the flattened grid
    random_values = np.random.choice(flattened_grid, size=num_values, replace=False)

    return random_values

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

def branching_model(alpha, k, h, pdiss,size,iterations,subsample_sort=1):
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
    #NOTE: I need different initialisation
    # Initialising the grid with random 0 and 1
    random_indices = np.random.choice(size * size, size=1, replace=False)
    aaa =[]
    # Assign the value of 1 to the random positions
    grid.flat[random_indices] = 1

    activ_prob: float = alpha * (1 / k)
    subsample_matrix = subsample(grid, subsample_sort)
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
                    #NOTE: I need something that excluded the original value
                    kk = random.sample(k_ij,k)
                         
                            
                    for ij in kk:
                            if  np.random.poisson(h) < activ_prob:
                                grid[ij[0],ij[1]] = 1
                                grid[i][j] = 0
                                k_points.append(k)
                                spikes.append(amount)
                                if subsample_matrix[i][j] == 1:
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
alpha = 0.99
k = 4
r = 2.4
pdiss = 0.001

# Run the branching model
grid,u,aval,num = branching_model(alpha, k, r, pdiss, 50, 20000)



def plotheatmap(u_k, k):
    plt.clf()

    plt.xlabel("x ")
    plt.ylabel("y ")

    plt.pcolormesh(u_k, cmap='hot', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.ax.set_title('Spikes',fontsize=12)



    return plt

def animate(k):
    plt = plotheatmap(u[k], k)    
    plt.title(f'Spikes at Frame {k}')
frames = 1000
anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=frames, repeat=False )
anim.save("branching_heat.gif")



plt.figure()
plt.plot(num)
plt.figure()
plt.hist(num, bins=30)
a =np.array(aval)




            
from collections import Counter

def count_avalanches_2(data):
    avalanche_count = 0
    avalanche_durations = []
    is_in_aval = [int(n) for n in range(len(data))]
    for t in range(len(data)):
      for i in range(100):
        if t+i < len(data) and data[t][1] == data[t+i][0] and data[t][2] == data[t+i][2] - 1:
           is_in_aval[t+i]=is_in_aval[t]
            

    
    avalanche_count = np.unique(is_in_aval)
    avalanche_durations=Counter(is_in_aval).values()
    return avalanche_count, avalanche_durations,is_in_aval

avalanche_count, avalanche_durations_O, is_in_aval = count_avalanches_2(a)

#print("Number of avalanches:", avalanche_count)
#print("Avalanche durations:", avalanche_durations_O)
from scipy.optimize import curve_fit
def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

target_func = func_powerlaw
plt.figure()
plt.plot(avalanche_durations_O)
plt.figure()
plt.hist(avalanche_durations_O, bins=30)
#sol2 = curve_fit(func_powerlaw, avalanche_count, avalanche_durations_O, p0 = np.asarray([-1,10**5,0]))

plt.figure()

bins_s=np.zeros(max(avalanche_durations_O)-1)
bins=[]
for i in range(1,max(avalanche_durations_O)):
    for j in range(len(avalanche_durations_O)):
        if i==list(avalanche_durations_O)[j]:
            bins_s[i-1]+=1
    bins.append(i)

    
plt.loglog( bins,bins_s)

    