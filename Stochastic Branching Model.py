import numpy as np
import matplotlib.pyplot as plt
import random

def choose_random_values(grid, num_values):
    # flatten the grid into a 1D array
    flattened_grid = grid.flatten()

    # Randomly choose num_values elements from the flattened grid
    random_values = np.random.choice(flattened_grid, size=num_values, replace=False)

    return random_values

def branching_model(alpha, k, r, pdiss,size):
    # Initialize the grid of neurons
    grid= np.zeros((size,size))

    #NOTE: I need different initialisation
    # Initialising the grid with random 0 and 1
    random_indices = np.random.choice(size * size, size=size, replace=False)

    # Assign the value of 1 to the random positions
    grid.flat[random_indices] = 1

    activ_prob: float = alpha * (1 / k)

    for _ in range(100):

        # Loop through each neuron in the grid
        for i in range(np.shape(grid)[0]):
            for j in range(np.shape(grid)[1]):

                # If neuron spikes:
                if grid[i][j] == 1:

                    n_i = []
                    n_j = []

                    arr = np.arange(0,size-1,1)

                    list = arr.tolist()

                    # Activate each postsynaptic neuron with probability p
                    n_i.append(random.sample(list,k))
                    #NOTE: I need something that excluded the original value
                    n_j.append(random.sample(list,k))

                    for ni in n_i[0]:
                        for nj in n_j[0]:
                            if np.random.random() < activ_prob:
                                grid[ni][nj] = 1
                            else:
                                grid[ni][nj] = 0

    return grid

# Set the parameters
alpha = 1
k = 4
r = 5
pdiss = 0.001

# Run the branching model
grid = branching_model(alpha, k, r, pdiss, 50)

# Print the resulting grid
print(grid)

plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()