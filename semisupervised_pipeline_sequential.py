import numpy as np
import os
import sys
import time

import create_modis
import extract_payload
import modis_l2
from cloud_mask import get_cloud_mask
from cloudsat import get_cloudsat_mask
from utils import all_invalid, contain_invalid, fill_all_channels

def semisupervised_pipeline_run(target_filepath, level2_dir, cloudsat_dir, save_dir, verbose=1):
    """
    :param target_filepath: the filepath of the radiance (MOD02) input file
    :param level2_dir: the root directory of l2 level files
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

    for dr in [save_dir_daylight, save_dir_night]:
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
    # TODO: check also if daylight or not https://michelanders.blogspot.com/2010/12/calulating-sunrise-and-sunset-in-python.html
    #t1 = time.time()
    #if all_invalid(np_swath[:2]):
    #    save_subdir = save_dir_night
    #    # all channels but visible ones
    #    fill_all_channels(np_swath[2:13])

    #else:
    #    save_subdir = save_dir_daylight
    #    # all channels but visible ones
    #    fill_all_channels(np_swath[:13})
    #t2 = time.time()

    #if verbose:
    #    print("Interpolation took {} s".format(t2-t1))

    # add in the L2 channels here
    # this includes only LWP, cloud optical depth atm, cloud top pressure
    # these can be filled with NaN, however as they are not being passed to the IRESNET, that is OK
    lwp, cod, ctp, cth = modis_l2.run(modis_files[0])

    # get cloud mask channel
    cm = get_cloud_mask(level2_dir, target_filepath)

    # get cloudsat labels channel
    # last two channels of np_swath correspond to Latitude and Longitude
    t1 = time.time()
    lm = get_cloudsat_mask(target_filepath, cloudsat_dir, np_swath[-2], np_swath[-1])
    t2 = time.time()

    if verbose:
        print("Cloudsat alignment took {} s".format(t2 - t1))

    # add the arrays to the end as separate channels
    np_swath = np.vstack([np_swath, lwp, cod, ctp, cth, cm, lm])
    assert np_swath.shape[0] == 21, "wrong number of channels"

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    swath_savepath_str = os.path.join(save_subdir, "swath", tail.replace(".hdf", ".npy"))
    np.save(swath_savepath_str, np_swath, allow_pickle=False)

    if verbose:
        print("swath {} saved".format(tail))

    # sample the swath for a selection of tiles and its associated metadata
    tiles, metadata = extract_payload.striding_tile_extract_from_file(np_swath, target_filepath, tile_size=3, stride=3)

    if verbose:
        print("tiles and metadata extracted from swath {}".format(tail))

    # create the save filepaths for the payload and metadata, and save the npys
    tiles_savepath_str = os.path.join(save_subdir, "tiles", tail.replace(".hdf", ".npy"))
    metadata_savepath_str = os.path.join(save_subdir, "metadata", tail.replace(".hdf", ".npy"))

    np.save(tiles_savepath_str, tiles, allow_pickle=False)
    np.save(metadata_savepath_str, metadata, allow_pickle=False)

    if verbose == (2 or 1):
        print("swath {} processed".format(tail))

# Hook for bash
if __name__ == "__main__":
    target_filepath = sys.argv[1]
    semisupervised_pipeline_run(target_filepath,
                                level2_dir="~/DATA/level_2/",
                                cloudsat_dir="~/DATA/cc_with_hours/",
                                save_dir="~/DATA/semisuper_sequential/",
                                verbose=1)
