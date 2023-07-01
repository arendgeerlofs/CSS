import matplotlib.pyplot as plt
import numpy as np

def plotheatmap(matrix):
    """
    Function that plots heatmap of spiking neurons in a matrix

    Input
    matrix: NxN matrix of neuron values

    Output
    plt: heatmap figure
    """
    plt.clf()

    plt.xlabel("x")
    plt.ylabel("y")
    # transform matrix to spiking and non-spiking neurons
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            if matrix[i][j] > 0:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0

    # create heatmap
    plt.pcolormesh(matrix, cmap=plt.cm.jet, vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.ax.set_title('Spikes',fontsize=12)

    return plt
