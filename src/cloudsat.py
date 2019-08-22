import datetime
import glob
import numpy as np
import os
import pickle
import random

from src.track_alignment import scalable_align, align

def get_month_day(day, year):
    """ Returns month and day given a day of a year"""

    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
    return dt.month, dt.day

def get_pickle_datetime(filename, year):
    
    pkl_time = os.path.basename(filename).replace(".pkl", "").split("_")
    pkl_month, pkl_day = int(pkl_time[0]), int(pkl_time[1])
    pkl_hour, pkl_minutes = int(pkl_time[2]), int(pkl_time[3])

    return datetime.datetime(year, pkl_month, pkl_day, pkl_hour, pkl_minutes)

def find_pickles_by_day(abs_day, year):
    """ returns list of pickle filenames of specified day, and of previous and following day """

    cloudsat_filenames = []

    for i in range(-1, 2):

        month, day = get_month_day(abs_day + i, year)

        # get all cloudsat pickles of that day
        str_month_day = "{}_{}_".format(month, day)
        cloudsat_filenames += glob.glob(os.path.join(cloudsat_dir, "{}*.pkl".format(str_month_day)))

    return cloudsat_filenames

def list_to_3d_array(list_labels):

    p = len(list_labels)
    array = np.zeros((8, p)) 

    for i, labels in enumerate(list_labels):
        for l in labels:

            # keep only cloud types (0 is non-determined or error)
            if l > 0:
                array[l-1][i] += 1

    return array

def get_cloudsat_info(l1_filename, cloudsat_dir):
    # MYD021KM.A2008003.1855.061.2018031033116.hdf l1
    # 1_3_18_15.pkl cloudsat

    time_info = l1_filename.split('MYD021KM.A')[1]
    year, abs_day = int(time_info[:4]), int(time_info[4:7])
    month, day = get_month_day(abs_day, year)
    hour, minutes = int(time_info[8:10]), int(time_info[10:12])

    swath_dt = datetime.datetime(year, month, day, hour, minutes)

    # get all candidate pickles
    cloudsat_filenames = find_pickles_by_day(abs_day, year)
    
    # collect all pickles before and after swath's time
    prev_candidates, foll_candidates = [], []

    for filename in cloudsat_filenames:

        dt = get_pickle_time(filename, year)
        
        if dt <= swath_dt:
            prev_candidates.append(dt)

        else:
            foll_candidates.append(dt)

    prev_dt = max(prev_candidates)
    foll_dt = min(foll_candidates)

    # load cloudsat pickle
    prev_filename = os.path.join(cloudsat_dir, "{}_{}_{}_{}.pkl".format(prev_dt.month, prev_dt.day, prev_dt.hour, prev_dt.minute))
    with open(prev_filename, "rb") as f:
        
        # pickle containing three lists, corresponding to latitude, longitude and label
        cloudsat_list = pickle.load(f)

    # if swath crosses over two cloudsat pickles, merge them
    if (foll_dt - swath_dt).seconds < 300:

        # load cloudsat pickle
        foll_filename = os.path.join(cloudsat_dir, "{}_{}_{}_{}.pkl".format(foll_dt.month, foll_dt.day, foll_dt.hour, foll_dt.minute))
        with open(foll_filename, "rb") as f:
            
            # concatenate the two pickles information
            cloudsat_list = [cloudsat_list[i] + values for i, values in pickle.load(f)]

    return cloudsat_list

def get_track_oi(track_points, latitudes, longitudes):

    min_lat = np.min(latitudes)
    max_lat = np.max(latitudes)

    min_lon = np.min(longitudes)
    max_lon = np.max(longitudes)
    
    return track_points[:, np.logical_and.reduce([[track_points[0] >= min_lat], [track_points[0] <= max_lat], [track_points[1] >= min_lon], [track_points[1] <= max_lon]]).squeeze()]

def find_track_range(track_points, latitudes, longitudes):

    i = random.randint(1, 2028)

    i_lat, i_lon = latitudes[i-1:i+1, :], longitudes[i-1:i+1, :]
    
    i_track_points = get_track_oi(track_points, i_lat, i_lon)
    
    i_mask = scalable_align(i_track_points, i_lat, i_lon)

    idx_nonzeros = np.where(np.sum(i_mask, 2) != 0)

    min_j, max_j = min(idx_nonzeros[1]), max(idx_nonzeros[1])

    return min_j - 100, max_j + 100

def get_cloudsat_mask(l1_filename, cloudsat_dir, latitudes, longitudes):

    cloudsat_list = get_cloudsat_info(l1_filename, cloudsat_dir)
    
    cloudsat = np.array([[c[0] for c in cloudsat_list[i]] for i in range(2)])
    cloudsat = np.vstack((cloudsat, list_to_3d_array(cloudsat_list[2])))   

    # focus around cloudsat track
    cs_range = find_track_range(cloudsat, latitudes, longitudes)
    lat, lon = latitudes[:, cs_range[0]:cs_range[1]], longitudes[:, cs_range[0]:cs_range[1]]

    track_points = get_track_oi(cloudsat, lat, lon)
    cloudsat_mask = scalable_align(track_points, lat, lon)

    # remove labels on egdes
    cloudsat_mask[0] = 0
    cloudsat_mask[-1] = 0

    print("retrieved", np.sum(cloudsat_mask > 0), "labels")
    
    # go back to initial swath size
    ext_cloudsat_mask = np.zeros((*(latitudes.shape), 8))
    ext_cloudsat_mask[:, cs_range[0]:cs_range[1], :] = cloudsat_mask

    return ext_cloudsat_mask.transpose(2, 0, 1).astype(np.uint8)    


if __name__ == "__main__":

    import sys
    
    import modis_level1

    target_filepath = sys.argv[1]
    head, tail = os.path.split(target_filepath)

    cloudsat_dir="../DATA/aqua-data/collocated_classes/cc_with_hours/"

    save_dir = "../DATA/aqua-data-processed/labelmask/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("geoloc found: {}".format(geoloc_filepath))

    # pull a numpy array from the hdfs
    np_swath = modis_level1.get_swath(target_filepath)

    lm = get_cloudsat_mask(target_filepath, cloudsat_dir, np_swath[-2], np_swath[-1])

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    savepath = os.path.join(save_dir, tail.replace(".hdf", ".npy"))
    np.save(savepath, lm, allow_pickle=False)
