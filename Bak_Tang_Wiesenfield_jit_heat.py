import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from numba import jit

@jit
def BTW(size, threshold, time_steps, labda, h):
    """
    """
    BTW = np.random.normal(0, 1, (size, size))
    BTW_new = np.zeros((size, size))
    BTW_spike_times = np.zeros((size, size))
    BTW_timed = []
    for k in range(time_steps):
        print("------------------")
        for i in range(np.shape(BTW)[0]):
            for j in range(np.shape(BTW)[1]):
                if j + 1 < 50 and j - 1 >= 0 and i + 1 < 50 and i - 1 >= 0:
                    if BTW[i][j] > threshold:
                        BTW_new[i][j] = BTW[i][j] - 4
                        BTW_spike_times[i][j] = k
                        print((i, j), k)
                    else:
                        BTW_new[i][j] = BTW[i][j] + 0.9* (labda * (k - BTW_spike_times[i-1][j]) + labda * (k - BTW_spike_times[i+1][j])
                                    + labda * (k - BTW_spike_times[i][j-1]) + labda * (k - BTW_spike_times[i][j+1])) + np.random.poisson(h)
        BTW = BTW_new.copy()
        BTW_timed.append(BTW)
    
    return BTW_timed

BTW(50, 0, 100, 0.001, 0.1)

def plotheatmap(u_k, k):
    plt.clf()

    plt.xlabel("x [inch]")
    plt.ylabel("y [inch]")

    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=4)
    cbar = plt.colorbar()
    cbar.ax.set_title('Spikes',fontsize=12)



    return plt



def animate(k):
    plt = plotheatmap(u[k], k)
    
    plt.title(f'Heat distribution at Frame {k}')
frames = 1000
u = BTW(50, 0, frames, 0.005, 0.5)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=frames, repeat=False )
anim.save("spike_heat.gif")


