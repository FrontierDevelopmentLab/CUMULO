import create_modis
from utils import fill_all_channels, contain_invalid
import extract_payload
import numpy as np
import os
import sys


def unsupervised_pipeline_run(target_filepath, save_dir, verbose=1):

    head, tail = os.path.split(target_filepath)

    save_dirs = [os.path.join(save_dir, "swath"), os.path.join(save_dir, "tiles"), os.path.join(save_dir, "metadata")]

    for dir_path in save_dirs:
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            if verbose:
                print("{} exists".format(dir_path))
            pass

    geoloc_filepath = create_modis.find_matching_geoloc_file(target_filepath)

    if verbose:
        print("geoloc found: {}".format(geoloc_filepath))

    modis_files = [target_filepath, geoloc_filepath]

    np_swath = create_modis.get_swath(modis_files)

    if verbose:
        print("swath {} loaded".format(tail))

    if verbose:
        print("swath shape: {}".format(np_swath.shape))

    fill_all_channels(np_swath)

    new_array = np.ma.masked_invalid(np_swath)

    if not contain_invalid(new_array):
        if verbose:
            print("swath {} interpolated".format(tail))
        pass
    else:
        raise ValueError("swath did not interpolate successfully")

    swath_savepath_str = os.path.join(save_dir, "swath", tail.replace(".hdf", ".npy"))
    np.save(swath_savepath_str, np_swath, allow_pickle=False)

    if verbose:
        print("swath {} saved".format(tail))

    tiles, metadata = extract_payload.random_tile_extract_from_file(np_swath, target_filepath, tile_size=3)

    if verbose:
        print("tiles and metadata extracted from swath {}".format(tail))

    tiles_savepath_str = os.path.join(save_dir, "tiles", tail.replace(".hdf", ".npy"))
    metadata_savepath_str = os.path.join(save_dir, "metadata", tail.replace(".hdf", ".npy"))

    np.save(tiles_savepath_str, tiles, allow_pickle=False)
    np.save(metadata_savepath_str, metadata, allow_pickle=False)

    if verbose == (2 or 1):
        print("swath {} processed".format(tail))


if __name__ == "__main__":
    target_filepath = sys.argv[1]
    unsupervised_pipeline_run(target_filepath, save_dir="../DATA/pipeline_output/190723_unsupervised_run_1", verbose=1)
