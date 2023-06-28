import numpy as np
import matplotlib.pyplot as plt
from Stochastic_Branching import branching_model, count_avalanches_SB
import matplotlib.animation as animation
from Heatmap import plotheatmap




#Define parameters
alpha = 0.95
k = 4
r = 2.1
pdiss = 0.001
size = 50
frames = 10


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

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=10, repeat=False )
anim.save("branching_heat.gif")

plt.figure()
plt.plot(num)
plt.figure()
plt.hist(num, bins=30)
avalanches = np.array(avalanches)

avalanche_count, avalanche_durations, is_in_aval = count_avalanches_SB(avalanches, int(2*max(num)))



plt.figure()
plt.plot(avalanche_durations)
plt.figure()
plt.hist(avalanche_durations, bins=30)

def bins(avalanche_durations):
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