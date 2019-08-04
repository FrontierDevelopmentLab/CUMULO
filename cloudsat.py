import datetime
import glob
import numpy as np
import os
import pickle

from align_track import scalable_align, align

def get_month_day(day, year):
    """ Returns month and day given a day of a year"""

    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
    return dt.month, dt.day

def get_cloudsat_filename(l1_filename, cloudsat_dir):
    # MYD021KM.A2008003.1855.061.2018031033116.hdf l1
    # 1_3_18_15.pkl cloudsat

    time_info = l1_filename.split('MYD021KM.A')[1]
    year, day = int(time_info[:4]), int(time_info[4:7])
    month, day = get_month_day(day, year)
    hour, minutes = int(time_info[8:10]), int(time_info[10:12])

    # get all cloudsat pickles of that day
    str_month_day = "{}_{}_".format(month, day)
    cloudsat_filenames = glob.glob(os.path.join(cloudsat_dir, "{}*.pkl".format(str_month_day)))
    
    # keep only the pkl with the corresponding time
    candidates = []
    for filename in cloudsat_filenames:

        pkl_time = os.path.basename(filename)[len(str_month_day):].replace(".pkl", "").split("_")
        pkl_hour, pkl_minutes = int(pkl_time[0]), int(pkl_time[1])

        if (pkl_hour, pkl_minutes) < (hour, minutes):
            candidates.append((pkl_hour, pkl_minutes))

    pkl_hour, pkl_minutes = max(candidates)

    return os.path.join(cloudsat_dir, "{}{}_{}.pkl".format(str_month_day, pkl_hour, pkl_minutes))

def get_interest_track(track_points, latitudes, longitudes):
    
    print(track_points.shape)
    min_lat = np.min(latitudes)
    max_lat = np.max(latitudes)

    min_lon = np.min(longitudes)
    max_lon = np.max(longitudes)

    return track_points[:, np.logical_and.reduce([[track_points[0] >= min_lat], [track_points[0] <= max_lat], [track_points[1] >= min_lon], [track_points[1] <= max_lon]]).squeeze()]

def list_to_3d_array(list_labels):

    p = len(list_labels)
    array = np.zeros((8, p)) 

    for i, labels in enumerate(list_labels):
        for l in labels:
            if l > 0:
                array[l-1][i] += 1

    return array


def get_cloudsat_mask(l1_filename, cloudsat_dir, latitudes, longitudes):

    cloudsat_filename = get_cloudsat_filename(l1_filename, cloudsat_dir)
    
    with open(cloudsat_filename, "rb") as f:
        
        # pickle containing three lists, corresponding to latitude, longitude and label
        cloudsat_list = pickle.load(f)

        # convert pickle to numpy array. The first two dims correspond to latitude and longitude coordinates, third dim corresponds to labels and may contain multiple values
        cloudsat = np.array([[c[0] for c in cloudsat_list[i]] for i in range(2)])
        cloudsat = np.vstack((cloudsat, list_to_3d_array(cloudsat_list[2])))    
    # focus only on central part of the swath
    
    cs_range = (300, 1000)
    lat, lon = latitudes[:, cs_range[0]:cs_range[1]].copy(), longitudes[:, cs_range[0]:cs_range[1]].copy()

    track_points = get_interest_track(cloudsat, lat, lon)
    cloudsat_mask = scalable_align(track_points, lat, lon)
    print("retrieved", np.sum(cloudsat_mask > 0), "labels")
    
    # go back to initial swath size
    ext_cloudsat_mask = np.zeros((*(latitudes.shape), 8))
    print(ext_cloudsat_mask.shape)
    ext_cloudsat_mask[:, cs_range[0]:cs_range[1], :] = cloudsat_mask

    print(np.min(ext_cloudsat_mask), np.max(ext_cloudsat_mask))

    return ext_cloudsat_mask    


if __name__ == "__main__":

    import sys
    
    import create_modis

    target_filepath = sys.argv[1]
    head, tail = os.path.split(target_filepath)

    cloudsat_dir="../DATA/aqua-data/collocated_classes/cc_with_hours/"

    save_dir = "./test-align/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # find a corresponding geolocational (MOD03) file for the provided radiance (MOD02) file
    geoloc_filepath = create_modis.find_matching_geoloc_file(target_filepath)

    print("geoloc found: {}".format(geoloc_filepath))

    # pull a numpy array from the hdfs, now that we have both radiance and geolocational files
    np_swath = create_modis.get_swath(target_filepath, geoloc_filepath)

    lm = get_cloudsat_mask(target_filepath, cloudsat_dir, np_swath[-2], np_swath[-1])

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    savepath = os.path.join(save_dir, tail.replace(".hdf", ".npy"))
    np.save(savepath, lm, allow_pickle=False)
