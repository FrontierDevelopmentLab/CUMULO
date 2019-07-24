from create_modis import get_swath_rgb
import numpy as np
from PIL import Image
import os
import sys
from utils  import fill_all_channels, contain_invalid
# from scipy.misc import toimage --depreciated, using PILlow

from PIL import Image
from create_modis import get_swath_rgb, find_matching_geoloc_file

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
    geoloc_filepath = find_matching_geoloc_file(radiance_filepath)

    if verbose:
        print("geoloc found: {}".format(geoloc_filepath))

    visual_swath = get_swath_rgb(radiance_filepath, geoloc_filepath)
    try:
        print(visual_swath.shape)
    except:
        raise ValueError("swath has no shape")
    
    png = Image.fromarray(visual_swath.astype(np.uint8), mode="RGB")

    #interpolate to remove NaN artefacts
    fill_all_channels(visual_swath)

    # checking if the interpolation is successful
    new_array = np.ma.masked_invalid(np_swath)
    if not contain_invalid(new_array):
        if verbose:
            print("swath {} interpolated".format(tail))
        pass
    else:
        raise ValueError("swath did not interpolate successfully")

    pil_loaded_visual_swath = Image.fromarray(visual_swath)

    save_filename = os.path.join(save_dir, basename.replace(".hdf", ".png"))
    png.save(save_filename)

    if verbose:
        print("swath {} processed".format(tail))


# Hook for pipe in
if __name__ == "__main__":
    target_filepath = sys.argv[1]
    save_swath_rbgs(target_filepath, save_dir="..DATA/pipeline_output/190723_png_extract", verbose=1)
