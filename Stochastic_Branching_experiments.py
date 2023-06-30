import numpy as np
import matplotlib.pyplot as plt
from Stochastic_Branching import branching_model, count_avalanches_SB
import matplotlib.animation as animation
from Heatmap import plotheatmap




#Define parameters
alpha = 0.95
k = 4
r = 0.1
pdiss = 0.001
size = 50
frames = 75000


# Run the branching model
grid, u, avalanches, num = branching_model(alpha, k, r, pdiss, size, frames)

def animate(k):
    """
    Function that is used for the animation of the time evolution of the model.
    Input
    k: current frame
    """
    plt = plotheatmap(u[k])
    plt.title(f'Heat distribution at Frame {k}')

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=500, repeat=False )
anim.save("branching_heat.gif")

#plot number of spikes over time and histogram
plt.figure()
plt.plot(num)
plt.figure()
plt.hist(num, bins=30)
avalanches = np.array(avalanches)

avalanche_count, avalanche_durations, is_in_aval = count_avalanches_SB(avalanches, int(2*max(num)))

#plot avalnche sizes

plt.figure()
plt.plot(avalanche_durations)
plt.figure()
plt.hist(avalanche_durations, bins=30)

def bins(avalanche_durations):
    """This function initializes an array bins_count with zeros, 
    where the length of the array is determined by the maximum duration minus 1. 
    It also creates an empty list bins to store the bin values.
    
    The function then iterates over the range of durations from 1 to the maximum duration. 
    For each duration, it counts the occurrences of that duration in the avalanche_durations 
    list and increments the corresponding bin count in the bins_count array.
    
    Input:
        avalanche_durations: list of durations corresponding to each avalanche.
    Output:
        bins: list of observed avalanche sizes
        bins_count: list of count of avalanche size of given size
    """
    bins_count = np.zeros(max(avalanche_durations)-1)
    bins = []
    for i in range(1,max(avalanche_durations)):
        for j in range(len(avalanche_durations)):
            if i == list(avalanche_durations)[j]:
                bins_count[i-1]+=1
        bins.append(i)
    return bins, bins_count


bins, bins_count = bins(avalanche_durations)
plt.figure()
plt.loglog(bins, bins_count)
plt.xlabel('s, avalanche size')
plt.ylabel('f(s), frequency in absolute counts')
plt.title('Frequency plot for Stochastic Branching')
