import datetime
import glob
import numpy as np
import os
import pickle

from pyhdf.SD import SD, SDC 
from pyhdf.HDF import HDF
from pyhdf.VS import VS

from src.track_alignment import get_track_oi, find_track_range, map_labels, scalable_align

def get_datetime(year, day, hour=0, minute=0, second=0):
    """ Returns month and day given a day of a year"""

    dt = datetime.datetime(year, 1, 1, hour, minute, second) + datetime.timedelta(days=day-1)
    return dt

def find_cloudsat_by_day(abs_day, year, cloudsat_dir):
    """ returns list of filenames of specified day, and of previous and following day """

    cloudsat_filenames = []

    for i in range(-1, 2):

        pattern = "{}{}{}*.hdf".format(year, "0" * (3 - len(str(abs_day + i))), abs_day + i)
        cloudsat_filenames += glob.glob(os.path.join(cloudsat_dir, pattern))

    return cloudsat_filenames

def find_matching_cloudsat_files(radiance_filename, cloudsat_dir):
    """
    :param radiance_filename: the filename for the radiance .hdf, demarcated with "MYD02".
    :return cloudsat_filenames: a list of paths to the corresponding cloudsat files (1 or 2 files)
    The time of the radiance file is used for selecting the cloudsat files: a MODIS swath is acquired every 5 minutes, while a CLOUDSAT granule is acquired every ~99 minutes. It can happen that a swath crosses over two granules. The filenames specify the starting time of the acquisition.
    CLOUDSAT filenames are in the format: AAAADDDHHMMSS_*.hdf
    """

    basename = os.path.basename(radiance_filename)
    
    time_info = radiance_filename.split('MYD021KM.A')[1]
    year, abs_day = int(time_info[:4]), int(time_info[4:7])
    hour, minutes = int(time_info[8:10]), int(time_info[10:12])

    swath_dt = get_datetime(year, abs_day, hour, minutes)

    cloudsat_filenames = find_cloudsat_by_day(abs_day, year, cloudsat_dir)

    # collect all granules before and after swath's time
    prev_candidates, foll_candidates = {}, {}

    for filename in cloudsat_filenames:
        
        cs_time_info = os.path.basename(filename)
        year, day, hour, minute, second = int(cs_time_info[:4]), int(cs_time_info[4:7]), int(cs_time_info[7:9]), int(cs_time_info[9:11]), int(cs_time_info[11:13])

        granule_dt = get_datetime(year, day, hour, minute, second)

        if granule_dt <= swath_dt and (swath_dt - granule_dt).seconds < 6000:
            prev_candidates[granule_dt] = filename

        elif (granule_dt - swath_dt).seconds < 300:
            foll_candidates[granule_dt] = filename

    prev_dt = max(prev_candidates.keys())
    
    # if swath crosses over two cloudsat granules, return both
    if len(foll_candidates.keys()) > 0:
        
        foll_dt = min(foll_candidates.keys())
        
        return prev_candidates[prev_dt], foll_candidates[foll_dt]
            
    return [prev_candidates[prev_dt]] 

def get_precip_flag(cloudsat_filenames, cloudsat_dir, verbose=0):

    all_flags = []

    for cloudsat_path in cloudsat_filenames:

        basename = os.path.basename(cloudsat_path)
        filename = glob.glob(os.path.join(cloudsat_dir, basename[4:7], basename[:11] + "*.hdf"))[0]
        
        f = HDF(cloudsat_path, SDC.READ) 
        vs = f.vstart() 
        
        vdata_precip = vs.attach('Precip_flag')
        precip = vdata_precip[:]
        
        if verbose:
            print("hdf information", vs.vdatainfo())
            print('Nb pixels: ', len(precip))
            print('Precip_flag values: ', np.unique(precip))

        all_flags += precip

        # close everything
        vdata_precip.detach()

        vs.end()
        f.close()
    
    return np.array(all_flags).flatten().astype(np.int8)

def get_coordinates(cloudsat_filenames, verbose=0):
    
    all_latitudes, all_longitudes = [], []

    for cloudsat_path in cloudsat_filenames:
        
        f = HDF(cloudsat_path, SDC.READ) 
        vs = f.vstart() 
        
        vdata_lat = vs.attach('Latitude')
        vdata_long = vs.attach('Longitude')

        latitudes = vdata_lat[:]
        longitudes = vdata_long[:]
        
        assert len(latitudes) == len(longitudes), "cloudsat hdf corrupted"
        
        if verbose:
            print("hdf information", vs.vdatainfo())
            print('Nb pixels: ', len(latitudes))
            print('Lat min, Lat max: ', min(latitudes), max(latitudes))
            print('Long min, Long max: ', min(longitudes), max(longitudes))

        all_latitudes += latitudes
        all_longitudes += longitudes

        # close everything
        vdata_lat.detach()
        vdata_long.detach()
        vs.end()
        f.close()
    
    return np.array(all_latitudes).flatten(), np.array(all_longitudes).flatten()

def get_layer_information(cloudsat_filenames, verbose=0):
    """ Returns
    CloudLayerType: -9: error, 0: non determined, 1-8 cloud types 
    CloudLayerBase: in km
    CloudLayerTop: in km
    CloudTypeQuality: valid range [0, 1]
    """
    
    all_types, all_bases, all_tops, all_qualities = [], [], [], []

    for cloudsat_path in cloudsat_filenames:

        sd = SD(cloudsat_path, SDC.READ)
        
        if verbose:
            # List available SDS datasets.
            print("hdf datasets:", sd.datasets())
        
        # get cloud types at each height
        all_types.append(sd.select('CloudLayerType').get())
        all_bases.append(sd.select('CloudLayerBase').get())
        all_tops.append(sd.select('CloudLayerTop').get())
        all_qualities.append(sd.select('CloudTypeQuality').get())

    layer_type = np.vstack(all_types)
    layer_base = np.vstack(all_bases)
    layer_top = np.vstack(all_tops)
    layer_type_quality = np.vstack(all_qualities)

    return layer_type.astype(np.int8), layer_base.astype(np.float16), layer_top.astype(np.float16), layer_type_quality.astype(np.float16)

def get_class_occurrences(layer_types):
    """ 
    Takes in a numpy.ndarray of size (nb_points, 10) describing for each point of the track the types of clouds
    identified at each of the 10 heights and returns a numpy.ndarray of size (nb_points, 8) counting the number 
    of times one of the 8 type of clouds was spotted vertically.
    The height information is then lost. 
    """
    
    occurrences = np.zeros((layer_types.shape[0], 8))
    
    for occ, labels in zip(occurrences, layer_types):
        
        for l in labels:
                
            # keep only cloud types (no 0 or -9)
            if l > 0:
                occ[l-1] += 1
    
    return occurrences    

def get_cloudsat_mask(l1_filename, cloudsat_lidar_dir, cloudsat_dir, swath_latitudes, swath_longitudes):

    cloudsat_filenames = find_matching_cloudsat_files(l1_filename, cloudsat_lidar_dir)
    cs_latitudes, cs_longitudes = get_coordinates(cloudsat_filenames)
    precip_flag = get_precip_flag(cloudsat_filenames, cloudsat_dir)
    layer_type, layer_base, layer_top, layer_type_quality = get_layer_information(cloudsat_filenames) 

    # focus around cloudsat track
    cs_range = find_track_range(cs_latitudes, cs_longitudes, swath_latitudes, swath_longitudes)
    lat, lon = swath_latitudes[:, cs_range[0]:cs_range[1]], swath_longitudes[:, cs_range[0]:cs_range[1]]

    toi_indices = get_track_oi(cs_latitudes, cs_longitudes, lat, lon)
    cs_latitudes, cs_longitudes, precip_flag, layer_type, layer_base, layer_top, layer_type_quality = cs_latitudes[toi_indices], cs_longitudes[toi_indices], precip_flag[toi_indices], layer_type[toi_indices], layer_base[toi_indices], layer_top[toi_indices], layer_type_quality[toi_indices]

    mapping = scalable_align(cs_latitudes, cs_longitudes, lat, lon)
    class_counts = get_class_occurrences(layer_type)
    cloudsat_mask = map_labels(mapping, class_counts, lat.shape, nb_classes=8)

    # remove labels on egdes
    cloudsat_mask[:10] = 0
    cloudsat_mask[:-11:-1] = 0

    print("retrieved", np.sum(cloudsat_mask > 0), "labels")
    
    # go back to initial swath size
    ext_cloudsat_mask = np.zeros((*(swath_latitudes.shape), 8))
    ext_cloudsat_mask[:, cs_range[0]:cs_range[1], :] = cloudsat_mask

    return ext_cloudsat_mask.transpose(2, 0, 1).astype(np.uint8), cs_range, mapping, layer_type, layer_base, layer_top, layer_type_quality, precip_flag    


if __name__ == "__main__":

    import sys
    
    import modis_level1

    target_filepath = sys.argv[1]
    head, tail = os.path.split(target_filepath)

    cloudsat_dir="../DATA/aqua-data/cloudsat_CC/"

    save_dir = "../DATA/aqua-data-processed/cloudsat/labelmasks/"
    save_dir_layer = "../DATA/aqua-data-processed/cloudsat/layers/"

    for d in [save_dir, save_dir_layer]:
        if not os.path.exists(d):
            os.makedirs(d)

    # pull a numpy array from the hdfs
    np_swath = modis_level1.get_swath(target_filepath)

    lm, cs_range, mapping, layer_type, layer_base, layer_top, layer_type_quality, precip_flag = get_cloudsat_mask(target_filepath, cloudsat_dir, np_swath[-2], np_swath[-1])

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    savepath = os.path.join(save_dir, tail.replace(".hdf", ".npy"))
    np.save(savepath, lm, allow_pickle=False)

    savepath = os.path.join(save_dir_layer, tail.replace(".hdf", ".npy"))
    cs_dict = {"width-range": cs_range, "mapping": mapping, "type-layer": layer_type, "base-layer": layer_base, "top-layer": layer_top, "type-quality": layer_type_quality, "precip-flag": precip_flag}
    np.save(savepath, cs_dict)