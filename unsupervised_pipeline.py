import create_modis
from utils import fill_all_channels, contain_invalid
import extract_payload
import numpy as np
import os
import sys


def unsupervised_pipeline_run(target_filepath, save_dir, verbose=1):
    """
    :param target_filepath: the filepath of the radiance (MOD02) input file
    :param save_dir:
    :param verbose: verbosity switch: 0 - silent, 1 - verbose, 2 - partial, only prints confirmation at end
    :return: none
    Wrapper for the full pipeline. Expects to find a corresponding MOD03 file in the same directory. Comments throughout
    """

    head, tail = os.path.split(target_filepath)

    # creating the save directories
    save_dirs = [os.path.join(save_dir, "swath"), os.path.join(save_dir, "tiles"), os.path.join(save_dir, "metadata")]

    for dir_path in save_dirs:
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            if verbose:
                print("{} exists".format(dir_path))
            pass

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
    fill_all_channels(np_swath)

    # checking if the interpolation is successful
    new_array = np.ma.masked_invalid(np_swath)
    if not contain_invalid(new_array):
        if verbose:
            print("swath {} interpolated".format(tail))
        pass
    else:
        raise ValueError("swath did not interpolate successfully")

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    swath_savepath_str = os.path.join(save_dir, "swath", tail.replace(".hdf", ".npy"))
    np.save(swath_savepath_str, np_swath, allow_pickle=False)

    if verbose:
        print("swath {} saved".format(tail))

    # sample the swath for a selection of tiles and its associated metadata
    tiles, metadata = extract_payload.random_tile_extract_from_file(np_swath, target_filepath, tile_size=3)

    if verbose:
        print("tiles and metadata extracted from swath {}".format(tail))

    # create the save filepaths for the payload and metadata, and save the npys
    tiles_savepath_str = os.path.join(save_dir, "tiles", tail.replace(".hdf", ".npy"))
    metadata_savepath_str = os.path.join(save_dir, "metadata", tail.replace(".hdf", ".npy"))

    np.save(tiles_savepath_str, tiles, allow_pickle=False)
    np.save(metadata_savepath_str, metadata, allow_pickle=False)

    if verbose == (2 or 1):
        print("swath {} processed".format(tail))


# Hook for bash
if __name__ == "__main__":
    target_filepath = sys.argv[1]
    unsupervised_pipeline_run(target_filepath, save_dir="../DATA/pipeline_output/190723_unsupervised_run_1", verbose=1)
