import numpy as np
import random
import os


def random_tile_extract_from_file(swath_array, file_path, tile_size=3):
    """
    :param swath_array: an input numpy array of a MODIS swath.
    :param file_path: The original filepath of the swath, used for naming
    :param tile_size: the size of the tile for sampling.
    :return: a list of two arrays: a 4-d payload array of sample,channel.w*h; and a metadata array of slice (w & h)
    This script will randomly sample down the swatch array that it inherits. Excluding padding, which varies depending
    on the tile size, the code iterates down the swath by pixel, randomly across the breadth of the swath for a tile.
    The mid-point of the swath where we expect labelled data in the future has been purposely avoided in the sampling.
    In addition to passing back sliced tiles across all channels, the script also returns the slice values themselves
    as metadata for future visualisation
    """
    _, tail = os.path.split(file_path)

    nullcheck = (lambda x: np.isnan(x).any())

    swath_bands, swath_length, swath_breadth = swath_array.shape

    filecheck_nans = nullcheck(swath_array)
    if filecheck_nans:
        print("WARNING: {} nan check failed".format(tail))

        non_nan_in_array = np.count_nonzero(~np.isnan(swath_array))
        elements_in_array = len(swath_array.flatten())

        print("{} is {}% complete".format(tail, (non_nan_in_array/elements_in_array)*100))

    offset = tile_size // 2
    offset_2 = offset

    if not tile_size % 2:
        offset_2 = offset + 1

    payload = []
    metadata = []

    for vertical_pixel in range((offset+1), swath_length-(offset+1)):

        tiles_in_band = []

        random_range = random.choice([(offset + 1, (swath_breadth//2)-(offset+1)),
                                      ((swath_breadth//2)+(offset+1),  swath_breadth-(offset+1))])

        random_horizontal_pixel = random.randint(*random_range)

        for band in range(swath_bands):

            tile = swath_array[band,
                               vertical_pixel - offset: vertical_pixel + offset_2 + 1,
                               random_horizontal_pixel - offset: random_horizontal_pixel + offset_2 + 1
                               ]

            tiles_in_band.append(tile)

        tile_metadata = [
            (vertical_pixel - offset, vertical_pixel + offset_2 + 1),
            (random_horizontal_pixel - offset, random_horizontal_pixel + offset_2 + 1)
        ]

        payload.append(tiles_in_band)
        metadata.append(tile_metadata)

    payload_array = np.stack(payload)

    return[payload_array, metadata]
