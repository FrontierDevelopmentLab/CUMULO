import numpy as np
import random

from scipy.stats import mode
from sklearn.metrics.pairwise import manhattan_distances

MAX_WIDTH, MAX_HEIGHT = 1354, 2030

def get_track_oi(cs_latitudes, cs_longitudes, swath_latitudes, swath_longitudes):

    max_lon, min_lon = np.max(swath_longitudes), np.min(swath_longitudes)
    max_lat, min_lat = np.max(swath_latitudes), np.min(swath_latitudes)
    
    return np.logical_and.reduce([[cs_latitudes >= min_lat], [cs_latitudes <= max_lat], [cs_longitudes >= min_lon], [cs_longitudes <= max_lon]]).squeeze()

def find_track_range(cs_latitudes, cs_longitudes, latitudes, longitudes):

    i = MAX_HEIGHT // 2

    i_lat, i_lon = latitudes[i-1:i+1, :], longitudes[i-1:i+1, :]
    
    i_indices = get_track_oi(cs_latitudes, cs_longitudes, i_lat, i_lon)
    
    i_mapping = scalable_align(cs_latitudes[i_indices], cs_longitudes[i_indices], i_lat, i_lon)

    min_j, max_j = min(i_mapping[1]), max(i_mapping[1])

    return max(0, min_j - 100), min(max_j + 100, MAX_WIDTH - 1)

def scalable_align(cs_lat, cs_lon, swath_lat, swath_lon):
    """  """
    (n, m) = swath_lat.shape

    swath_points = np.stack((swath_lat.flatten(), swath_lon.flatten())).T
    track_points = np.stack((cs_lat, cs_lon), axis=1)
  
    dist = manhattan_distances(swath_points, track_points)
    mapping = np.unravel_index(np.argmin(dist, axis=0), (n, m))

    return mapping

def map_labels(mapping, labels, shape):

    labelmask = np.zeros((*shape, labels.shape[1]))

    for i, l in enumerate(labels):
        labelmask[mapping[0][i], mapping[1][i]] += l

    return labelmask

def map_and_reduce(mapping, track, swath, width_range, reduce_method="mode"):
    """ modify swath!!! 
        As multiple points from track can be mapped to the same point in swath, take the most common value.
    """

    shape = swath[:, width_range[0]:width_range[1]].shape

    # cannot use np.ndarray as number of track points mapped to same swath point is unknown a priori
    mapped_values = {}

    for i, values in enumerate(track):

        try:
            mapped_values[mapping[0][i], mapping[1][i]].append(values)
        except KeyError:
            mapped_values[mapping[0][i], mapping[1][i]] = [values]

    # reduce by mode
    concat_axis = 1
    if len(shape) < 3:
        concat_axis = 0

    for (i, j), values in mapped_values.items():

        # remove values from edges which are luckily to have been oversampled
        if i > 9 and i < shape[0] - 10:
    
            values = np.stack(values, concat_axis)

            swath[:, width_range[0]:width_range[1]][i, j] = mode(values, axis=concat_axis)[0].flatten()

if __name__ == "__main__":

    # example
    test_lat = np.array([[8., 10., 12.],
                     [8.1, 10., 12.2],
                     [8.6, 10.9, 12.1],
                     [9.6, 11.1, 13.1],
                     [10.6, 11.9, 13.5]])

    test_lon = np.array([[10.,  20.,  30.], 
                         [10.1, 21.1, 33.3], 
                         [12.9, 22.9, 34.4], 
                         [14.2, 26.1, 35.5], 
                         [15.4, 28.9, 36.6]])

    test_track = np.array([[8.7, 9.1, 10.1, 13.7], [11.1, 18.4, 39.1, 45.9], [1, 3, 7, 6]])

    mapping = scalable_align(test_track[0], test_track[1], test_lat, test_lon)
    labels = map_labels(mapping, np.array(test_track[2])[:, None], (5, 3))

    print(labels)
