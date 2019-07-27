import numpy as np
import math
import pdb

# fast euclidean distance computation: https://stackoverflow.com/questions/37794849/efficient-and-precise-calculation-of-the-euclidean-distance
def eudis5(v1, v2):
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    #dist = math.sqrt(sum(dist))
    return dist
    
def align(track_points, swath_lat, swath_lon):
    p = track_points.shape[0]
    n = swath_lat.shape[0]
    m = swath_lat.shape[1]
    swath_lat = swath_lat.reshape((n,m,1))
    swath_lon = swath_lon.reshape((n,m,1))
    labels = np.zeros((n,m,8,p))
    # change track points to numpys
    L  = track_points[:,0].reshape((1,1,p))
    LA = track_points[:,1].reshape((1,1,p))
    LO = track_points[:,2].reshape((1,1,p))
    LA_dists = (LA - swath_lat)**2
    LO_dists = (LO - swath_lon)**2
    both = np.sqrt(LA_dists + LO_dists)

    both_indsR, both_indsC = np.unravel_index(np.argmin(both.reshape(n*m,p),axis=0), (n,m))
    #temp = np.ravel_multi_index((LA_indsR, LA_indsC, np.arange(p)), (n,m,p))
    #d_indLA = LA_dists.flatten()[temp]
    #LA_locs  = np.argmin(LA_dists,1)
    
    #LO_indsR, LO_indsC = np.unravel_index(np.argmin(LO_dists.reshape(n*m,p),axis=0), (n,m))
    #temp = np.ravel_multi_index((LO_indsR, LO_indsC, np.arange(p)), (n,m,p))
    #d_indLO = LO_dists.flatten()[temp]

    #nearest_inds = np.argmin((d_indLA, d_indLO), axis=0)

    
    #LO_locs  = np.argmin(LO_dists,1)
    #both_RC = np.array((both_indsR, both_indsC))
    #unique, counts = np.unique(both_RC, return_counts=True, axis=1)
    locs = np.concatenate((both_indsR[np.newaxis,:],both_indsC[np.newaxis,:],L.reshape((1,p)),np.arange(p).reshape((1,p))),0)
    locs = locs.astype(int)
    inds = np.ravel_multi_index(locs, (n,m,8,p))
    labels[locs[0,:],locs[1,:],locs[2,:],locs[3,:]] = 1
    labels = np.argmax(np.sum(labels,axis=3),axis=2)
    return labels

if __name__ == "__name__":

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
        
    points   = np.array([[1, 8.7, 11.1], 
                         [3, 9.1, 18.4],
                         [7, 10.1,39.1],
                         [6, 13.7,45.9]])

    labels = align(points, test_lat, test_lon)