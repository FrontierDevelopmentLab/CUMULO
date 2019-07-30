import numpy as np
import os
import sys
import time
import glob
import pdb

import create_modis
import extract_payload
import modis_l2
from cloud_mask import get_cloud_mask
from cloudsat import get_cloudsat_mask
from utils import all_invalid, contain_invalid, fill_all_channels


def semisupervised_pipeline_run(target_filepath, level2_dir, cloudmask_dir, cloudsat_dir, save_dir, verbose=1):
    """
    :param target_filepath: the filepath of the radiance (MOD02) input file
    :param level2_dir: the root directory of l2 level files
    :param cloudmask_dir: the root directory to cloud mask files
    :param cloudsat_dir: the root directory of cloudsat pkl files
    :param save_dir:
    :param verbose: verbosity switch: 0 - silent, 1 - verbose, 2 - partial, only prints confirmation at end
    :return: none
    Wrapper for the full pipeline. Expects to find a corresponding MOD03 file in the same directory. Comments throughout
    """

    head, tail = os.path.split(target_filepath)

    # creating the save directories
    save_dir_swath = os.path.join(save_dir, "swath")
    save_dir_daylight = os.path.join(save_dir_swath, "daylight")
    save_dir_night = os.path.join(save_dir_swath, "night")
    save_dir_fucked = os.path.join(save_dir_swath, "fucked")

    for dr in [save_dir_daylight, save_dir_night, save_dir_fucked]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    # find a corresponding geolocational (MOD03) file for the provided radiance (MOD02) file
    geoloc_filepath = create_modis.find_matching_geoloc_file(target_filepath)

    if verbose:
        print("geoloc found: {}".format(geoloc_filepath))

    # pull a numpy array from the hdfs, now that we have both radiance and geolocational files
    modis_files = [target_filepath, geoloc_filepath]
    np_swath = create_modis.get_swath(modis_files)

    if verbose:
        print("swath {} loaded".format(tail))
        print("swath shape: {}".format(np_swath.shape))

    # as some bands have artefacts, we need to interpolate the missing data - time intensive
    # check if visible channels contain NaNs
    # TODO: check also if daylight or not
    #  https://michelanders.blogspot.com/2010/12/calulating-sunrise-and-sunset-in-python.html
    t1 = time.time()
    try:
        if all_invalid(np_swath[:2]):
            save_subdir = save_dir_night
            # all channels but visible ones
            fill_all_channels(np_swath[2:13])

        else:
            save_subdir = save_dir_daylight
            fill_all_channels(np_swath[:13])

    except ValueError:
        save_subdir = save_dir_fucked
    
    t2 = time.time()

    if verbose:
            print("Interpolation took {} s".format(t2-t1))

    # add in the L2 channels here
    # this includes only LWP, cloud optical depth atm, cloud top pressure
    # these can be filled with NaN, however as they are not being passed to the IRESNET, that is OK
    l2_channels = modis_l2.run(target_filepath, level2_dir)
    
    if verbose:
        print("Level2 channels loaded")

    # get cloud mask channel
    cm = get_cloud_mask(cloudmask_dir, target_filepath)

    if verbose:
        print("Cloud mask loaded")

    # get cloudsat labels channel
    # last two channels of np_swath correspond to Latitude and Longitude
    t1 = time.time()
    # lm = get_cloudsat_mask(target_filepath, cloudsat_dir, np_swath[-2], np_swath[-1])
    parts = tail.split(".")
    year_day_part = parts[1]
    time_part = parts[2]
    
    lm_glob_query = "CC.{}.{}.npy".format(year_day_part, time_part)
    matching_cloud_mask = glob.glob(os.path.join(cloudsat_dir, lm_glob_query))
    try:
        lm = np.load(matching_cloud_mask[0])
    except IndexError:
        save_subdir = save_dir_fucked
        print("file {} has no matching cloudmask".format(tail))
    t2 = time.time()

    if verbose:
        print("Cloudsat alignment took {} s".format(t2 - t1))

    # add the arrays to the end as separate channels
    print(np_swath.shape, l2_channels.shape, cm.shape, lm.shape)
    print(np.sum(lm != 0))
    np_swath = np.vstack([np_swath, l2_channels, cm[None, ], lm[None, ]])

    assert np_swath.shape[0] == 20, "wrong number of channels"

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    swath_savepath_str = os.path.join(save_subdir, tail.replace(".hdf", ".npy"))
    np.save(swath_savepath_str, np_swath, allow_pickle=False)

    if verbose:
        print("swath {} saved".format(tail))

    # sample the swath for a selection of tiles and its associated metadata
    label_tiles, nonlabel_tiles, label_metadata, nonlabel_metadata = \
        extract_payload.extract_labels_and_cloud_tiles(np_swath, target_filepath, tile_size=3, stride=3)

    if verbose:
        print("tiles and metadata extracted from swath {}".format(tail))

    label_tiles_savepath_str = os.path.join(save_subdir, "label", "tiles")
    label_metadata_savepath_str = os.path.join(save_subdir, "label", "metadata")

    nonlabel_tiles_savepath_str = os.path.join(save_subdir, "nonlabel", "tiles")
    nonlabel_metadata_savepath_str = os.path.join(save_subdir, "nonlabel", "metadata")

    # create the save filepaths for the payload and metadata, and save the npys
    for dr in [label_tiles_savepath_str,
               label_metadata_savepath_str,
               nonlabel_tiles_savepath_str,
               nonlabel_metadata_savepath_str]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    np.save(os.path.join(label_tiles_savepath_str, tail.replace(".hdf", ".npy")),
            label_tiles, allow_pickle=False)
    np.save(os.path.join(label_metadata_savepath_str, tail.replace(".hdf", ".npy")),
            label_metadata, allow_pickle=False)
    np.save(os.path.join(nonlabel_tiles_savepath_str, tail.replace(".hdf", ".npy")),
            nonlabel_tiles, allow_pickle=False)
    np.save(os.path.join(nonlabel_metadata_savepath_str, tail.replace(".hdf", ".npy")),
            nonlabel_metadata, allow_pickle=False)

    if verbose == (2 or 1):
        print("swath {} processed".format(tail))


# Hook for bash
if __name__ == "__main__":
    target_filepath = sys.argv[1]
    semisupervised_pipeline_run(target_filepath,
                                level2_dir="../DATA/aqua-data/level_2/",
                                cloudmask_dir="../DATA/aqua-data/cloud_mask/",
                                cloudsat_dir="../DATA/aqua-data/labelled_arrays/",
                                save_dir="../DATA/semisuper-sequential/",
                                verbose=1)
