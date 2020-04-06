import glob
import numpy as np
import os
import pickle

from pyhdf.SD import SD, SDC 
from pyhdf.HDF import HDF
from pyhdf.VS import VS

from src.track_alignment import get_track_oi, find_track_range, map_labels, scalable_align
from src.utils import get_datetime, get_file_time_info

def find_cloudsat_by_day(abs_day, year, cloudsat_dir):
    """ returns list of filenames of specified day, and of previous and following day """

    cloudsat_filenames = []

    for i in range(-1, 2):
        
        cur_day = abs_day + i

        pattern = "{}{}/{}{}{}*.hdf".format("0" * (3 - len(str(cur_day))), cur_day, year, "0" * (3 - len(str(cur_day))), cur_day)
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

    year, abs_day, hour, minutes = get_file_time_info(basename)
    year, abs_day, hour, minutes = int(year), int(abs_day), int(hour), int(minutes)

    swath_dt = get_datetime(year, abs_day, hour, minutes)

    cloudsat_filenames = find_cloudsat_by_day(abs_day, year, cloudsat_dir)

    # collect all granules before and after swath's time
    prev_candidates, foll_candidates = {}, {}

    for filename in cloudsat_filenames:
        
        cs_time_info = os.path.basename(filename)
        year, day, hour, minute, second = int(cs_time_info[:4]), int(cs_time_info[4:7]), int(cs_time_info[7:9]), int(cs_time_info[9:11]), int(cs_time_info[11:13])

        granule_dt = get_datetime(year, day, hour, minute, second)

        if granule_dt <= swath_dt and (swath_dt - granule_dt).total_seconds() < 6000:
            prev_candidates[granule_dt] = filename

        elif granule_dt >= swath_dt and (granule_dt - swath_dt).total_seconds() < 300:
            foll_candidates[granule_dt] = filename

    prev_dt = max(prev_candidates.keys())
    
    # if swath crosses over two cloudsat granules, return both
    if len(foll_candidates.keys()) > 0:
        
        foll_dt = min(foll_candidates.keys())
        
        return prev_candidates[prev_dt], foll_candidates[foll_dt]
            
    return [prev_candidates[prev_dt]] 

def get_precip_flag(cloudsat_filenames, cloudsat_dir=None, verbose=0):

    all_flags = []

    for cloudsat_path in cloudsat_filenames:
        
        # if precipitation information is stored in another file
        if cloudsat_dir is not None:
            basename = os.path.basename(cloudsat_path)
            filename = glob.glob(os.path.join(cloudsat_dir, basename[:11] + "*.hdf"))[0]
        else:
            filename = cloudsat_path
        
        f = HDF(filename, SDC.READ) 
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

def get_layer_information(cloudsat_filenames, get_quality=True, verbose=0):
    """ Returns
    CloudLayerType: -9: error, 0: non determined, 1-8 cloud types 
    CloudLayerBase: in km
    CloudLayerTop: in km
    CloudTypeQuality: valid range [0, 1]; if <get_quality>
    """
    
    all_info = {'CloudLayerType': [], 'CloudLayerBase': [], 'CloudLayerTop': []}

    if get_quality:
        all_info['CloudTypeQuality'] = []

    for cloudsat_path in cloudsat_filenames:

        sd = SD(cloudsat_path, SDC.READ)
        
        if verbose:
            # List available SDS datasets.
            print("hdf datasets:", sd.datasets())
        
        # get cloud types at each height
        for key, value in all_info.items():
            value.append(sd.select(key).get())

    for key, value in all_info.items():
        value = np.vstack(value)

        if key == 'CloudLayerType':
            all_info[key] = value.astype(np.int8)
        else:
            all_info[key] = value.astype(np.float16)

    if not get_quality:
        all_info['CloudTypeQuality'] = None

    return all_info

def get_class_occurrences(layer_types):
    """ 
    Takes in a numpy.ndarray of size (nb_points, 10) describing for each point of the track the types of clouds identified at each of the 10 heights and returns a numpy.ndarray of size (nb_points, 8) counting the number of times one of the 8 type of clouds was spotted vertically.
    The height information is then lost. 
    """
    
    occurrences = np.zeros((layer_types.shape[0], 8))
    
    for occ, labels in zip(occurrences, layer_types):
        
        for l in labels:
                
            # keep only cloud types (no 0 or -9)
            if l > 0:
                occ[l-1] += 1
    
    return occurrences    

def get_cloudsat_mask(l1_filename, cloudsat_lidar_dir, cloudsat_dir, swath_latitudes, swath_longitudes, map_labels=True):

    # retrieve cloudsat files content
    if cloudsat_lidar_dir is None:

        cloudsat_filenames = find_matching_cloudsat_files(l1_filename, cloudsat_dir)
        precip_flag = get_precip_flag(cloudsat_filenames)
        # LayerTypeQuality not available in CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00 files
        layer_info = get_layer_information(cloudsat_filenames, get_quality=False) 

    else:

        cloudsat_filenames = find_matching_cloudsat_files(l1_filename, cloudsat_lidar_dir)

        if cloudsat_dir is not None:
            precip_flag = get_precip_flag(cloudsat_filenames, cloudsat_dir)
        else:
            precip_flag = None
            
        layer_info = get_layer_information(cloudsat_filenames, get_quality=True) 

    layer_info['PrecipFlag'] = precip_flag

    # focus around cloudsat track
    cs_latitudes, cs_longitudes = get_coordinates(cloudsat_filenames)

    cs_range = find_track_range(cs_latitudes, cs_longitudes, swath_latitudes, swath_longitudes)
    lat, lon = swath_latitudes[:, cs_range[0]:cs_range[1]], swath_longitudes[:, cs_range[0]:cs_range[1]]

    toi_indices = get_track_oi(cs_latitudes, cs_longitudes, lat, lon)
    cs_latitudes, cs_longitudes = cs_latitudes[toi_indices], cs_longitudes[toi_indices]

    mapping = scalable_align(cs_latitudes, cs_longitudes, lat, lon)

    for key, value in layer_info.items():
        
        if value is not None:
            layer_info[key] = value[toi_indices]

    if map_labels:
        class_counts = get_class_occurrences(layer_type)
        cloudsat_mask = map_labels(mapping, class_counts, lat.shape)

        # remove labels on egdes
        cloudsat_mask[:10] = 0
        cloudsat_mask[:-11:-1] = 0

        print("retrieved", np.sum(cloudsat_mask > 0), "labels")
        
        # go back to initial swath size
        ext_cloudsat_mask = np.zeros((*(swath_latitudes.shape), 8))
        ext_cloudsat_mask[:, cs_range[0]:cs_range[1], :] = cloudsat_mask

        return ext_cloudsat_mask.transpose(2, 0, 1).astype(np.uint8), cs_range, mapping, layer_info
    
    else:

        return cs_range, mapping, layer_info

if __name__ == "__main__":

    import sys
    
    import modis_level1

    target_filepath = sys.argv[1]
    head, tail = os.path.split(target_filepath)

    cloudsat_lidar_dir=None
    cloudsat_dir = "."

    save_dir = "../test/aqua-data-processed/cloudsat/labelmasks/"
    save_dir_layer = "../test/aqua-data-processed/cloudsat/layers/"

    for d in [save_dir, save_dir_layer]:
        if not os.path.exists(d):
            os.makedirs(d)

    # pull a numpy array from the hdfs
    np_swath = modis_level1.get_swath(target_filepath)

    cs_range, mapping, layer_info = get_cloudsat_mask(target_filepath, cloudsat_lidar_dir, cloudsat_dir, np_swath[-2], np_swath[-1], map_labels=False)

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    savepath = os.path.join(save_dir, tail.replace(".hdf", ".npy"))
    np.save(savepath, lm, allow_pickle=False)

    savepath = os.path.join(save_dir_layer, tail.replace(".hdf", ".npy"))
    cs_dict = {"width-range": cs_range, "mapping": mapping}
    cs_dict.update(layer_info)

    np.save(savepath, cs_dict)
