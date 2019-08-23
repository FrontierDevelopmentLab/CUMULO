import numpy as np

from scipy import interpolate

def all_invalid(array, tol=5e-2):
    """ Checks if 3d array contains all invalid values.
        :param tol: tollerance ratio of invalid values 
    """

    masked_array = np.ma.masked_invalid(array)

    c, cols, rows = array.shape

    return np.sum(masked_array.mask) >= c * cols * rows * tol 

def contain_invalid(masked_array):
    """Checks to see if the array contain any 1s, which would indicate NaNs in the swath."""

    return np.sum(masked_array.mask) > 0

# ------------------------------------------------------------------------------ INTERPOLATION

def fill_channel(masked_array, xx, yy, method="nearest"):
    """ 
        Inplace function: it fills all invalid valued by spatial interpolation
        :param swath (numpy.array): array of size (nb_channels, height, width) 
        :param method (string): method for the interpolation. Check scipy.interpolate.griddata for possible methods
        :return: list of channels that have been filled
    """

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
        :return: list of channels that have been filled or were already full
    """

    swath_shape = swath.shape

    x, y = np.arange(0, swath_shape[2]), np.arange(0, swath_shape[1])
    xx, yy = np.meshgrid(x, y)

    full_channels = []

    for i, ch_array in enumerate(swath):

        masked_array = np.ma.masked_invalid(ch_array)

        if contain_invalid(masked_array):
            
            try:
                inter = fill_channel(masked_array, xx, yy, method)
                swath[i] = inter

                full_channels.append(i)
            
            except:
                pass

        else:

            full_channels.append(i)

    return full_channels

if __name__ == "__main__":

    # kinda test all invalid

    all_inv_array = np.array([[[np.NaN] * 10] * 4] * 3)
    assert all_invalid(all_inv_array)

    partial_inv_array = np.zeros((3, 7, 9))
    partial_inv_array[0] = np.NaN
    assert not all_invalid(partial_inv_array)

