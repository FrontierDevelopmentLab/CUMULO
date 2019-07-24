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


def striding_tile_extract_from_file(swath_array, file_path, tile_size=3, stride=1):
    """
    :param swath_array: input numpy array from MODIS
    :param file_path: the original filepath, for verbose functions
    :param tile_size: size of the tile selected from within the image
    :param stride: horizontal steps between images
    :return: a list of two arrays: a 4-d payload array of sample,channel.w*h; and a metadata array of slice (w & h)

    This script will systematically sample down the swatch array that it inherits. Excluding padding, which varies
    depending on the tile size, the code iterates down the swath by pixel, and strides across the breadth of the
    swath for a tile. The mid-point of the swath where we expect labelled data in the future has been purposely
    avoided in the sampling. In addition to passing back sliced tiles across all channels, the script also returns
    the slice values themselves as metadata for future visualisation

    """

    _, tail = os.path.split(file_path)

    nullcheck = (lambda x: np.isnan(x).any())

    swath_bands, swath_length, swath_breadth = swath_array.shape

    filecheck_nans = nullcheck(swath_array)
    if filecheck_nans:
        print("WARNING: {} nan check failed".format(tail))

        non_nan_in_array = np.count_nonzero(~np.isnan(swath_array))
        elements_in_array = len(swath_array.flatten())

        print("{} is {}% complete".format(tail, (non_nan_in_array / elements_in_array) * 100))

    _, tail = os.path.split(file_path)

    nullcheck = (lambda x: np.isnan(x).any())

    filecheck_nans = nullcheck(swath_array)
    if filecheck_nans:
        print("WARNING: {} nan check failed".format(tail))

        non_nan_in_array = np.count_nonzero(~np.isnan(swath_array))
        elements_in_array = len(swath_array.flatten())

        print("{} is {}% complete".format(tail, (non_nan_in_array / elements_in_array) * 100))

    offset = tile_size // 2
    offset_2 = offset

    if not tile_size % 2:
        offset_2 = offset + 1

    payload = []
    metadata = []

    lower_breadth_range = np.arange(start=(offset + 1),
                                    stop=(swath_breadth // 2 - (offset + 1)),
                                    step=stride)

    upper_breadth_range = np.arange(start=(swath_breadth // 2 + (offset + 1)),
                                    stop=(swath_breadth - (offset + 1)),
                                    step=stride)

    horizontal_pixels = np.append(lower_breadth_range, upper_breadth_range)

    vertical_pixels = np.arange(start=(offset + 1),
                                stop=(swath_length - (offset + 1)),
                                step=stride)

    centre_of_tile_position = []
    for vertical_pixel in vertical_pixels:
        for horizontal_pixel in horizontal_pixels:
            coordinates = (vertical_pixel, horizontal_pixel)
            centre_of_tile_position.append(coordinates)

    for co_ord in centre_of_tile_position:
        vertical_pos = co_ord[0]
        horizontal_pos = co_ord[1]

        bands_in_tile = []
        for band in range(swath_bands):
            tile = swath_array[band,
                               vertical_pos - offset: vertical_pos + offset_2 + 1,
                               horizontal_pos - offset: horizontal_pos + offset_2 + 1
                               ]

            bands_in_tile.append(tile)

        tile_metadata = [
            (vertical_pos - offset, vertical_pos + offset_2 + 1),
            (horizontal_pos - offset, horizontal_pos + offset_2 + 1)]

        metadata.append(tile_metadata)
        payload.append(bands_in_tile)

    payload_array = np.stack(payload)

    return [payload_array, metadata]
