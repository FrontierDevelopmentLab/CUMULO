import numpy as np

from scipy import interpolate

def fill_channel(matrix, xx, yy, method="nearest"):

    array = np.ma.masked_invalid(matrix)
        
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]

    new_array = array[~array.mask]

    inter = interpolate.griddata((x1, y1), new_array.ravel(), (xx, yy), method=method, fill_value=True)

    return inter

def fill_all_channels(swath, method="nearest"):
    """ 
        param swath (np.array) : array of size (nb_channels, height, width) 
    """

    swath_shape = swath.shape

    x, y = np.arange(0, swath_shape[2]), np.arange(0, swath_shape[1])
    xx, yy = np.meshgrid(x, y)

    filled_swath = []

    for ch_matrix in swath:

        inter = fill_channel(ch_matrix, xx, yy, method)

        filled_swath.append(inter)

    filled_swath = np.stack(filled_swath)

    return filled_swath