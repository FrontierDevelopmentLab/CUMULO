import numpy as np

from sklearn.metrics.pairwise import manhattan_distances

def scalable_align(track, swath_lat, swath_lon):
    """  """
    (n, m) = swath_lat.shape
    labels = np.zeros((n, m, 8))

    swath_points = np.stack((swath_lat.flatten(), swath_lon.flatten())).T
    track_points = track[:2].T

    L = track[2:]

    p = L.shape[1]
  
    dist = manhattan_distances(swath_points, track_points)
    indices = np.unravel_index(np.argmin(dist, axis=0), (n, m))
    
    for i in range(p):
        labels[indices[0][i], indices[1][i]] += L[:, i]

    return labels

def align(track_points, swath_lat, swath_lon):
    """ Euclidean distance """
    p = track_points.shape[1]
    n = swath_lat.shape[0]
    m = swath_lat.shape[1]
    swath_lat = swath_lat.reshape((n,m,1))
    swath_lon = swath_lon.reshape((n,m,1))
    labels = np.zeros((n,m))
    # change track points to numpys
    L  = track_points[2].reshape((1,p))
    LA = track_points[0].reshape((1,1,p))
    LO = track_points[1].reshape((1,1,p))
    LA_dists = (LA - swath_lat)**2
    LO_dists = (LO - swath_lon)**2
    both = np.sqrt(LA_dists + LO_dists)

    both_indsR, both_indsC = np.unravel_index(np.argmin(both.reshape(n*m,p),axis=0), (n,m))

    locs = np.concatenate((both_indsR[np.newaxis,:],both_indsC[np.newaxis,:]),0)
    inds = np.ravel_multi_index(locs, (n,m))
    labels[locs[0,:],locs[1,:]] = L

    return labels.astype(int)

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

    labels = align(test_track, test_lat, test_lon)

    print(labels)

    labels = scalable_align(test_track, test_lat, test_lon)

    print(labels)
