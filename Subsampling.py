import numpy as np

def subsample(matrix, sort=1):
    """
    Function that computes subsampling matrix based on 6 different types of
    subsampling given in paper by Prieseman et al. 

    Input
    matrix: BTW matrix which you want to subsample
    sort: type of subsampling

    Output
    sample: matrix in which 1's indicate neurons in the subsample and 0's neurons not in subsample
    """
    # If no subsampling return all ones
    if sort == 0:
        return np.ones(np.shape(matrix))
    sample = np.zeros(np.shape(matrix))
    size = 0
    dist = 0
    # random subsampling based on size
    if sort <= 2:
        if sort == 1:
            size = 64
        else:
            size = 100
        i = 0
        # add randomly chosen neurons to subsample
        while i < size:
            index = np.random.randint(np.shape(matrix)[0], size = 2)
            if sample[index[0]][index[1]] != 1:
                sample[index[0]][index[1]] = 1
                i += 1
    # add neurons based on grid and distance between neurons
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