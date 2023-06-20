import matplotlib.pyplot as plt

def plotheatmap(u_k):
    plt.clf()

    plt.xlabel("x [inch]")
    plt.ylabel("y [inch]")

    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=4)
    cbar = plt.colorbar()
    cbar.ax.set_title('Spikes',fontsize=12)

    return plt
