import matplotlib.pyplot as plt
import numpy as np

def plotheatmap(u_k):
    plt.clf()

    plt.xlabel("x [inch]")
    plt.ylabel("y [inch]")
    for i in range(np.shape(u_k)[0]):
        for j in range(np.shape(u_k)[1]):
            if u_k[i][j] > 0:
                u_k[i][j] = 1
            else:
                u_k[i][j] = 0


    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.ax.set_title('Spikes',fontsize=12)

    return plt
