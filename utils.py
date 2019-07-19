import numpy as np

from scipy import interpolate

def contain_invalid(masked_matrix):

    return np.sum(masked_matrix.mask) > 0


def fill_channel(masked_matrix, xx, yy, method="nearest"):
        
    x1 = xx[~masked_matrix.mask]
    y1 = yy[~masked_matrix.mask]

    new_mask = masked_matrix[~masked_matrix.mask]

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

        masked_matrix = np.ma.masked_invalid(ch_matrix)

        if contain_invalid:
        
            inter = fill_channel(mask, xx, yy, method)
            filled_swath.append(inter)

        else:

            filled_swath.append(ch_matrix)

    filled_swath = np.stack(filled_swath)

    return filled_swath