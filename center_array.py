import numpy as np

def center_array(array, xy): #numpy array, NN head position
    og = np.copy(array)
    array.fill(0)
    array.resize(((2 * array.shape[0]) -1, (2 * array.shape[1]) -1), refcheck = False)
    array[0:og.shape[0],0:og.shape[1]] = og
    array = np.roll(array, (og.shape[0]-1)-xy[1], 0)
    array = np.roll(array, (og.shape[1]-1)-xy[0], 1)
    return array
