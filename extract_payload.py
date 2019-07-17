import pandas as pd
import numpy as np
import random
import os


def process_directory(file_directory, payload_path, metadata_path, tile_size=3):


def random_tile_extract_from_file(file_in, payload_path, metadata_path, tile_size=3):
    filename = file_in.split("/")[-1]
    filename = filename[:-4]

    swath_array = pd.read_pickle(file_in)

    swath_bands, swath_length, swath_breadth = swath_array.shape

    if not tile_size % 2:
        raise ValueError("Only odd-sized tile sizes accepted.")

    offset = tile_size // 2

    payload = []
    metadata = []

    for vertical_pixel in range(offset, swath_length-offset):

        tiles_in_band = []
        metadata_in_band = []

        random_horizontal_pixel = random.randint(offset+1, swath_breadth-(offset+1))

        for band in range(swath_bands):

            tile = swath_array[band,
                               vertical_pixel-offset : vertical_pixel+offset+1,
                               random_horizontal_pixel-offset : random_horizontal_pixel+offset+1]

            tile_metadata = ["{}".format(band),
                             "{}:{}".format(vertical_pixel-offset, vertical_pixel+offset+1),
                             "{}:{}".format(random_horizontal_pixel-offset, random_horizontal_pixel+offset+1)]

            tiles_in_band.append(tile)
            metadata_in_band.append(tile_metadata)

        payload.append(tiles_in_band)
        metadata.append(metadata_in_band)

    payload_array = np.stack(payload)
    payload_array = np.transpose(payload_array, (1, 0, 2, 3))

    payload_path = os.path.join(payload_path, "payload_{}".format(filename))
    metadata_path = os.path.join(metadata_path, "metadata_{}".format(filename))

    np.save(payload_path, payload_array, allow_pickle=False)
    np.save(metadata_path, metadata, allow_pickle=False)
