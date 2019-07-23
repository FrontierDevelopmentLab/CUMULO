import numpy as np
import os
import sys

from scipy.misc import toimage

from create_modis import get_swath_rgb

def save_swath_rbgs(radiance_filepath, save_dir, verbose=1):
    """
    :param radiance_filepath: the filepath of the radiance (MOD02) input file
    :param save_dir:
    :param verbose: verbosity switch: 0 - silent, 1 - verbose, 2 - partial, only prints confirmation at end
    :return: none
    Generate and save RBG channels of the given MODIS file. Expects to find a corresponding MOD03 file in the same directory. Comments throughout
    """

    basename = os.path.basename(radiance_filepath)

    # creating the save subdirectory
    save_dir = os.path.join(save_dir, "visual")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # find a corresponding geolocational (MOD03) file for the provided radiance (MOD02) file
    geoloc_filepath = create_modis.find_matching_geoloc_file(radiance_filepath)

    if verbose:
        print("geoloc found: {}".format(geoloc_filepath))

    visual_swath = get_swath_rgb(radiance_filepath, geoloc_filepath)

    save_filename = os.path.join(save_dir, basename.replace(".hdf", ".png"))
    toimage(visual_swath, cmin=0.0, cmax=255.0).save(save_filename)

    if verbose != 0:
        print("swath {} processed".format(tail))


# Hook for bash
if __name__ == "__main__":
    target_filepath = sys.argv[1]
    save_swath_rbgs(target_filepath, save_dir="./test", verbose=1)
