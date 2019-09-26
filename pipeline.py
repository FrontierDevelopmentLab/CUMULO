import numpy as np
import os
import time

from PIL import Image

import src.cloudsat
import src.interpolation
import src.modis_level1
import src.modis_level2
import src.tile_extraction

def extract_full_swath(target_filepath, level2_dir, cloudmask_dir, cloudsat_lidar_dir, cloudsat_dir, save_dir, verbose=1):
    """
    :param target_filepath: the filepath of the radiance (MYD02) input file
    :param level2_dir: the root directory of l2 level files
    :param cloudmask_dir: the root directory to cloud mask files
    :param cloudsat_dir: the root directory of cloudsat-lidar files
    :param cloudsat_dir: the root directory of cloudsat files
    :param save_dir:
    :param verbose: verbosity switch: 0 - silent, 1 - verbose, 2 - partial, only prints confirmation at end
    :return: none
    Expects to find a corresponding MYD03 file in the same directory. Comments throughout
    """

    _, tail = os.path.split(target_filepath)

    # creating the save directories
    save_dir_daylight = os.path.join(save_dir, "daylight")
    save_dir_night = os.path.join(save_dir, "night")
    save_dir_fucked = os.path.join(save_dir, "corrupt")

    for dr in [save_dir_daylight, save_dir_night, save_dir_fucked]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    # pull a numpy array from the hdfs
    np_swath = src.modis_level1.get_swath(target_filepath)

    if verbose:
        print("swath {} loaded".format(tail))

    # as some bands have artefacts, we need to interpolate the missing data - time intensive
    t1 = time.time()
    
    filled_ch_idx = src.interpolation.fill_all_channels(np_swath)  
    
    t2 = time.time()

    if verbose:
        print("Interpolation took {} s".format(t2-t1))
        print("Channels", filled_ch_idx, "are now full")

    # if all channels were filled
    if len(filled_ch_idx) == 15:
        save_subdir = save_dir_daylight

    # if all but visible channels were filled
    elif filled_ch_idx == list(range(2, 7)) + list(range(8, 15)):
        save_subdir = save_dir_night

    else:
        save_subdir = save_dir_fucked

    # pull L2 channels here
    l2_channels = src.modis_level2.get_channels(target_filepath, level2_dir)
    
    if verbose:
        print("Level2 channels loaded")

    # pull cloud mask channel
    cm = src.modis_level2.get_cloud_mask(target_filepath, cloudmask_dir)

    if verbose:
        print("Cloud mask loaded")

    # get cloudsat labels channel - time intensive
    t1 = time.time()

    try:

        lm, *additional_info = src.cloudsat.get_cloudsat_mask(target_filepath, cloudsat_lidar_dir, cloudsat_dir, np_swath[-2], np_swath[-1])
        np_swath = np.vstack([np_swath, l2_channels, cm[None, ], lm])

    except Exception as e:

        np_swath = np.vstack([np_swath, l2_channels, cm[None, ]])
        print("file {} has no matching labelmask: {}".format(tail, e))

    t2 = time.time()

    if verbose:
        print("Cloudsat alignment took {} s".format(t2 - t1))

    # cast swath values to float
    np_swath = np_swath.astype(np.float16)

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    swath_savepath_str = os.path.join(save_subdir, tail.replace(".hdf", ".npy"))
    np.save(swath_savepath_str, np_swath, allow_pickle=False)

    if verbose:
        print("swath saved as {}".format(swath_savepath_str))
    
    try:
        layer_info_savepath = os.path.join(save_subdir, "layer-info")
    
        if not os.path.exists(layer_info_savepath):
            os.makedirs(layer_info_savepath)

        cs_dict = {"width-range": additional_info[0], "mapping": additional_info[1], "type-layer": additional_info[2], "base-layer": additional_info[3], "top-layer": additional_info[4], "type-quality": additional_info[5], "precip-flag": additional_info[6]}
        np.save(os.path.join(layer_info_savepath, tail.replace(".hdf", ".npy")), cs_dict)

    except:
        pass

    return np_swath, save_subdir, tail

def extract_tiles_from_swath(np_swath, swath_name, save_dir, tile_size=3, stride=3, verbose=1):
    # sample the swath for a selection of tiles and its associated metadata
    try: 
        label_tiles, nonlabel_tiles, label_metadata, nonlabel_metadata = src.tile_extraction.sample_labelled_and_unlabelled_tiles(np_swath, tile_size=tile_size)

    except ValueError as e:
        print("Tiles failed to extract.", str(e))
        exit(0)

    if verbose > 0:
        print("{} tiles extracted from swath {}".format(len(label_tiles) + len(nonlabel_tiles), swath_name))

    label_tiles_savepath_str = os.path.join(save_dir, "label", "tiles")
    label_metadata_savepath_str = os.path.join(save_dir, "label", "metadata")

    nonlabel_tiles_savepath_str = os.path.join(save_dir, "nonlabel", "tiles")
    nonlabel_metadata_savepath_str = os.path.join(save_dir, "nonlabel", "metadata")

    # create the save filepaths for the payload and metadata, and save the npys
    for dr in [label_tiles_savepath_str, label_metadata_savepath_str, nonlabel_tiles_savepath_str, nonlabel_metadata_savepath_str]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    filename_npy = swath_name.replace(".hdf", ".npy")

    np.save(os.path.join(label_tiles_savepath_str, filename_npy), label_tiles, allow_pickle=False)
    np.save(os.path.join(label_metadata_savepath_str, filename_npy), label_metadata, allow_pickle=False)
    np.save(os.path.join(nonlabel_tiles_savepath_str, filename_npy), nonlabel_tiles, allow_pickle=False)
    np.save(os.path.join(nonlabel_metadata_savepath_str, filename_npy), nonlabel_metadata, allow_pickle=False)

    # save_tiles_separately(label_tiles, swath_name, os.path.join(save_dir, "label"))
    # save_tiles_separately(nonlabel_tiles, swath_name, os.path.join(save_dir, "nonlabel"))

def save_tiles_separately(tiles, swath_name, save_dir, tile_size=3):

    save_dir = os.path.join(save_dir, "all-tiles")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, tile in enumerate(tiles):
        
        np.save(os.path.join(save_dir, "{}-{}.npy".format(swath_name.replace(".hdf", ""), i)), tile)

def extract_swath_rbg(radiance_filepath, save_dir, verbose=1):
    """
    :param radiance_filepath: the filepath of the radiance (MYD02) input file
    :param save_dir:
    :param verbose: verbosity switch: 0 - silent, 1 - verbose, 2 - partial, only prints confirmation at end
    :return: none
    Generate and save RBG channels of the given MYDIS file. Expects to find a corresponding MYD03 file in the same directory. Comments throughout
    """

    basename = os.path.basename(radiance_filepath)

    # creating the save subdirectory
    save_dir = os.path.join(save_dir, "rgb")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    visual_swath = src.modis_level1.get_swath_rgb(radiance_filepath)
    
    #interpolate to remove NaN artefacts
    filled_ch_idx = src.interpolation.fill_all_channels(visual_swath)

    if len(filled_ch_idx) == 3:

        pil_loaded_visual_swath = Image.fromarray(visual_swath.transpose(1, 2, 0).astype(np.uint8), mode="RGB")

        save_filename = os.path.join(save_dir, basename.replace(".hdf", ".png"))
        pil_loaded_visual_swath.save(save_filename)

        if verbose > 0:
            print("RGB channels saved as {}".format(save_filename))

    else:
        print("Failed to interpolate RGB channels of", basename)

# Hook for bash
if __name__ == "__main__":

    import sys
    target_filepath = sys.argv[2]
    save_dir = sys.argv[1]
    
    root_dir = "/mnt/disks/disk1/"
    # extract training channels, validation channels, cloud mask, class occurences if provided
    np_swath, save_subdir, swath_name = extract_full_swath(target_filepath,
                                level2_dir=root_dir+"aqua-data/level_2",
                                cloudmask_dir=root_dir+"aqua-data/cloud_mask",
                                cloudsat_lidar_dir=root_dir+"aqua-data/cloudsat_CC/",
                                cloudsat_dir=root_dir+"aqua-data/cloudsat_CC/2008/",
                                save_dir=save_dir,
                                verbose=1)
    
    # extract visible channels for visualization purposes
    extract_swath_rbg(target_filepath, save_subdir, verbose=1)

    # extract tiles for Machine Learning purposes
    if np_swath.shape != (33, 2030, 1354):
        print("Failed to extract tiles: tiles are extracted only from swaths with label mask", np_swath.shape)
        exit(0)

    if "fucked" in save_subdir:
        print("Failed to extract tiles: tiles are extracted only from swaths with fully interpolated non-visible channels")
        exit(0)

    extract_tiles_from_swath(np_swath, swath_name, save_subdir)

