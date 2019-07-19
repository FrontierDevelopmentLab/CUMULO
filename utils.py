import numpy as np

from scipy import interpolate

def contain_invalid(masked_array):

    return np.sum(masked_array.mask) > 0

def fill_channel(masked_array, xx, yy, method="nearest"):
    
    x1 = xx[~masked_array.mask]
    y1 = yy[~masked_array.mask]

    new_mask = masked_array[~masked_array.mask]

    inter = interpolate.griddata((x1, y1), new_mask.ravel(), (xx, yy), method=method, fill_value=True)

    return inter

def fill_all_channels(swath, method="nearest"):
    """ 
        Inplace function: it fills all invalid valued by spatial interpolation a channel at a time
        :param swath (numpy.array): array of size (nb_channels, height, width) 
        :param method (string): method for the interpolation. Check scipy.interpolate.griddata for possible methods
        :return: list of channels that have been filled
    """

    swath_shape = swath.shape

    x, y = np.arange(0, swath_shape[2]), np.arange(0, swath_shape[1])
    xx, yy = np.meshgrid(x, y)

    filled_channels = []

    for i, ch_array in enumerate(swath):

        masked_array = np.ma.masked_invalid(ch_array)

        if contain_invalid(masked_array):
            
            inter = fill_channel(masked_array, xx, yy, method)
            swath[i] = inter

            filled_channels.append(i)

    return filled_channels