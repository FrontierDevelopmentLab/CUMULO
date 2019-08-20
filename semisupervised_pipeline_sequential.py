import numpy as np
import os
import sys
import time

import cloudsat
import interpolation
import modis_level1
import modis_level2
import tile_extraction

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

    _, tail = os.path.split(target_filepath)

    # creating the save directories
    save_dir_daylight = os.path.join(save_dir, "daylight")
    save_dir_night = os.path.join(save_dir, "night")
    save_dir_fucked = os.path.join(save_dir, "fucked")

    for dr in [save_dir_daylight, save_dir_night, save_dir_fucked]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    # pull a numpy array from the hdfs
    np_swath = modis_level1.get_swath(target_filepath)

    if verbose:
        print("swath {} loaded".format(tail))

    # as some bands have artefacts, we need to interpolate the missing data - time intensive
    t1 = time.time()
    try:
        if interpolation.all_invalid(np_swath[:2]):
            save_subdir = save_dir_night
            # interpolate all channels but visible ones
            interpolation.fill_all_channels(np_swath[2:13])

        else:
            save_subdir = save_dir_daylight
            interpolation.fill_all_channels(np_swath[:13])

    except ValueError:
        save_subdir = save_dir_fucked
    
    t2 = time.time()

    if verbose:
        print("Interpolation took {} s".format(t2-t1))

    # pull L2 channels here
    # this includes only LWP, cloud optical depth, cloud top pressure in this order
    l2_channels = modis_level2.get_lwp_cod_ctp(target_filepath, level2_dir)
    
    if verbose:
        print("Level2 channels loaded")

    # pull cloud mask channel
    cm = modis_level2.get_cloud_mask(target_filepath, cloudmask_dir)

    if verbose:
        print("Cloud mask loaded")

    # get cloudsat labels channel - time intensive
    t1 = time.time()

    try:

        lm = cloudsat.get_cloudsat_mask(target_filepath, cloudsat_dir, np_swath[-2], np_swath[-1])
        np_swath = np.vstack([np_swath, l2_channels, cm[None, ], lm])

    except:

        np_swath = np.vstack([np_swath, l2_channels, cm[None, ]])
        print("file {} has no matching labelmask".format(tail))

    t2 = time.time()

    if verbose:
        print("Cloudsat alignment took {} s".format(t2 - t1))

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    swath_savepath_str = os.path.join(save_subdir, tail.replace(".hdf", ".npy"))
    np.save(swath_savepath_str, np_swath, allow_pickle=False)

    if verbose:
        print("swath {} saved".format(tail))

    # sample the swath for a selection of tiles and its associated metadata
    try: 
        label_tiles, nonlabel_tiles, label_metadata, nonlabel_metadata = \
        tile_extraction.extract_labels_and_cloud_tiles(np_swath,
                target_filepath, tile_size=3, stride=3)
    except ValueError as e:
        print("Tiles failed to extract.", str(e))
        exit(0)

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
                                level2_dir="/mnt/disks/disk11/aqua-data/level_2",
                                cloudmask_dir="/mnt/disks/disk11/aqua-data/cloudmask",
                                cloudsat_dir="/mnt/disks/disk11/aqua-data/CC/cc_with_hours/",
                                save_dir="/mnt/disks/disk11/190805_run2_month3",
                                verbose=1)
