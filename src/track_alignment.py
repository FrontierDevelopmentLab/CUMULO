import numpy as np

from sklearn.metrics.pairwise import manhattan_distances

MAX_WIDTH, MAX_HEIGHT = 1354, 2030

def get_track_oi(cs_latitudes, cs_longitudes, swath_latitudes, swath_longitudes):

    max_lon, min_lon = np.max(swath_longitudes), np.min(swath_longitudes)
    max_lat, min_lat = np.max(swath_latitudes), np.min(swath_latitudes)
    
    return np.logical_and.reduce([[cs_latitudes >= min_lat], [cs_latitudes <= max_lat], [cs_longitudes >= min_lon], [cs_longitudes <= max_lon]]).squeeze()

def find_track_range(cs_latitudes, cs_longitudes, latitudes, longitudes):

    i = random.randint(1, MAX_HEIGHT - 2)

    i_lat, i_lon = latitudes[i-1:i+1, :], longitudes[i-1:i+1, :]
    
    i_indices = get_track_oi(cs_latitudes, cs_longitudes, i_lat, i_lon)
    
    i_mask = scalable_align(i_indices, i_lat, i_lon)

    idx_nonzeros = np.where(np.sum(i_mask, 2) != 0)

    min_j, max_j = min(idx_nonzeros[1]), max(idx_nonzeros[1])

    return min_j - 100, max_j + 100

def scalable_align(cs_lat, cs_lon, swath_lat, swath_lon):
    """  """
    (n, m) = swath_lat.shape

    swath_points = np.stack((swath_lat.flatten(), swath_lon.flatten())).T
    track_points = np.stack((cs_lat, cs_lon), axis=1)
  
    dist = manhattan_distances(swath_points, track_points)
    mapping = np.unravel_index(np.argmin(dist, axis=0), (n, m))

    return mapping

def map_labels(mapping, labels, shape, nb_classes=8):

    labelmask = np.zeros((*shape, labels.shape[1]))

    for i, l in enumerate(labels):
        labelmask[mapping[0][i], mapping[1][i]] += l

    return labelmask

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
