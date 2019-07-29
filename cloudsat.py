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

    min_lat = np.min(latitudes)
    max_lat = np.max(latitudes)

    min_lon = np.min(longitudes)
    max_lon = np.max(longitudes)

    return track_points[:, np.logical_and.reduce([[track_points[0] >= min_lat], [track_points[0] <= max_lat], [track_points[1] >= min_lon], [track_points[1] <= max_lon]]).squeeze()]

def get_cloudsat_mask(l1_filename, cloudsat_dir, latitudes, longitudes):

    cloudsat_filename = get_cloudsat_filename(l1_filename, cloudsat_dir)
    
    with open(cloudsat_filename, "rb") as f:
        
        # pickle containing three lists, corresponding to latitude, longitude and label
        cloudsat_list = pickle.load(f)

        # convert pickle to numpy array. The first two dims correspond to latitude and longitude coordinates, third dim corresponds to labels and may contain multiple values
        # TODO: keep all labels
        cloudsat = np.array([[c[0] for c in cloudsat_list[i]] for i in range(3)])
        # cloudsat.vstack([cloudsat_list[2]])
    # focus only on central part of the swath
    
    cs_range = (950, 1150)
    lat, lon = latitudes[cs_range[0]:cs_range[1]].copy(), longitudes[cs_range[0]:cs_range[1]].copy()

    track_points = get_interest_track(cloudsat, latitudes, longitudes)
    cloudsat_mask = scalable_align(track_points, lat, lon)
    print(np.sum(track_points[2] != 0), np.sum(cloudsat_mask != 0))
    
    # go back to initial swath size
    ext_cloudsat_mask = np.zeros(latitudes.shape)
    ext_cloudsat_mask[cs_range[0]:cs_range[1], :] = cloudsat_mask

    return ext_cloudsat_mask    
