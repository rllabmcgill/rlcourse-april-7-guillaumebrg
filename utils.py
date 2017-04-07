__author__ = 'Guillaume'

import numpy as np

def one_hot(idx, length):
    """Turn an integer number (or a vector of int numbers), into
    a one-hot vector (or a matrix whose rows are one-hot vectors)."""
    if np.ndim(idx) == 0:
        idx = np.array([idx])
    out = np.zeros((idx.shape[0], length), "float32")
    out[range(idx.shape[0]), idx] = 1
    return out

def preprocess_grid(grid):
    """Turn a 2D grid into a 3-channel image: one chanel for each kind
    of object (agent, obstacles, goal)"""
    # Add the batch axis & channel axis
    reshaped_grid = grid[None,None,:,:]
    # Stack and return as a 'float32' array
    return np.concatenate((reshaped_grid==1, reshaped_grid==2, reshaped_grid==3), axis=1).astype("float32")

def preprocess_grids(grids):
    """Same as 'preprocess_grid' but for several grids at the same time.
    No need to add the batch axis."""
    # Add the channel axis
    reshaped_grids = grids[:, None, :, :]
    # Stack and return as a 'float32' array
    return np.concatenate((reshaped_grids==1, reshaped_grids==2, reshaped_grids==3), axis=1).astype("float32")

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def epsilon_greedy(q, epsilon):
    if np.random.rand()<epsilon:
        A = np.argmax(q)
    else:
        A = np.random.randint(0, q.shape[0])
    return A