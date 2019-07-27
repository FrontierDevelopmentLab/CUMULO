import datetime
import glob
import numpy as np
import os
import pickle

from align_track import align

def get_month_day(day, year):
    """ Returns month and day given a day of a year"""

    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day)
    return dt.month, dt.day

def get_cloudsat_filename(l1_filename, cloudsat_dir):
    # MYD021KM.A2008003.1855.061.2018031033116.hdf l1
    # 1_3_18_15.pkl cloudsat

    time_info = l1_filename.split('MYD021KM.A')[1]
    year, day = time_info[:4], time_info[4:7]
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
            candidates.append(pkl_hour, pkl_minutes)

    pkl_hour, pkl_minutes = max(candidates)

    return os.path.join(cloudsat_dir, "{}{}_{}_{}.pkl".format(str_month_day, pkl_hour, pkl_minutes))

def get_cloudsat_mask(l1_filename, cloudsat_dir, latitudes, longitudes):

    cloudsat_filename = get_cloudsat_filename(l1_filename)

    with open glob.glob(cloudsat_filename) as f:
        
        # pickle containing three lists, corresponding to latitude, longitude and label
        cloudsat_list = pickle.load(f)

        # convert pickle to numpy array. The first two dims correspond to latitude and longitude coordinates, third dim corresponds to labels and may contain multiple values
        cloudsat = np.array([[c[0] for c in cloudsat_list[i]] for i in range(2)])
        cloudsat.vstack([cloudsat_list[2]])

    cloudsat_mask = align(cloudsat, latitudes, longitudes)

    return cloudsat_mask    