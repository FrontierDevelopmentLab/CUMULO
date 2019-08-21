import numpy as np
import os
import random

def get_sampling_mask(mask_shape=(2030, 1350), tile_size=3):
    """ returns a mask of allowed centers for the tiles to be sampled. The center of an even size tile is considered to be the point at the position (size // 2 + 1, size // 2 + 1) within the tile.
    """
    mask = np.ones(mask_shape, dtype=np.uint8)

    offset = tile_size // 2
    offset_2 = offset

    if not tile_size % 2:
        offset_2 -= 1

    # must not sample tile centers in the borders, so that tiles keep to required shape
    mask[:, :offset] = 0
    mask[:, -offset_2:] = 0
    mask[:offset, :] = 0
    mask[-offset_2:, :] = 0

    return mask

def get_label_mask(class_channels):
    """ given the class occurences channels, returns a 2d array marking as 1 the labelled pixels and as 0 the unlabelled ones."""

    labelmask = np.sum(class_channels, axis=0)
    labelmask[labelmask > 0] = 1

    return labelmask.astype(np.uint8) 

def sample_cloudy_unlabelled_tiles(swath_array, cloud_mask, label_mask, number_of_tiles, tile_size=3):
    """
    :param swath_array: input numpy array from MODIS of size (nb_channels, w, h)
    :param cloud_mask: 2d array of zise (w, h) marking the cloudy pixels 
    :param label_mask: 2d array of zise (w, h) marking the labelled pixels 
    :param number_of_tiles: the number of tiles to sample. It is reset to maximal number of tiles that can be sampled, if bigger
    :param tile_size: size of the tile selected from within the image
    :return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles; and a list of tuples ((w1, w2), (h1, h2)) with the relative positions of the sampled tiles withing the swath
    The script will use a cloud_mask channel to mask away all non-cloudy data and a label_mask channel to mask away all labelled data. The script will then randomly select a number of tiles (:param number of tiles) from the cloudy areas that are unlabeled.
    """

    # mask not to sample outside the swath
    allowed_pixels = get_sampling_mask(swath_array.shape[1:], tile_size)

    # combine the three masks, tile centers will be sampled from the cloudy and unlabelled pixels that are not in the borders of the swath
    allowed_pixels = np.logical_and.reduce(allowed_pixels, cloud_mask, ~label_mask)
    allowed_pixels_idx = np.where(allowed_pixels == 1)
    allowed_pixels_idx = list(zip(*allowed_pixels_idx))

    number_of_tiles = min(number_of_tiles, len(allowed_pixels_idx))

    # sample without replacement
    tile_centers = np.random.choice(allowed_pixels_idx, number_of_tiles, False)
    
    _, swath_length, swath_breadth = swath_array.shape

    for coord in tile_centers:
        vertical_pos = coord[1]
        horizontal_pos = coord[0]

        tile = swath_array[:,
                           horizontal_pos - offset: horizontal_pos + offset_2 + 1,
                           vertical_pos - offset: vertical_pos + offset_2 + 1
                          ]

        tile_metadata = [
            (horizontal_pos - offset, horizontal_pos + offset_2 + 1),
            (vertical_pos - offset, vertical_pos + offset_2 + 1)]

        metadata.append(tile_metadata)        
        payload.append(tile)

    payload_array = np.stack(payload)

    return payload_array, metadata


def extract_label_tiles(swath_array, cloud_mask, label_mask, tile_size=3):
    """
    :param swath_array: input swath, WITH labels as the last channel
    :param tile_size: the size of the channels
    :return: nested list of extracted tile and metadata
    """

    _, tail = os.path.split(file_path)

    offset = tile_size // 2
    offset_2 = offset

    if not tile_size % 2:
        offset_2 = offset + 1

    _, swath_length, _ = swath_array.shape

    label_indexes = np.where(np.logical_and(np.sum(swath_array[-8:], 0) > 0, swath_array[-9]))

    vertical_pos = label_indexes[1]
    horizontal_pos = label_indexes[0]

    payload = []
    metadata = []

    for i in range(len(vertical_pos)):

        if horizontal_pos[i] < (offset):
            continue

        if horizontal_pos[i] > (swath_length - (offset_2)):
            continue

        bands_in_tile = []

        tile = swath_array[:,
                           horizontal_pos[i] - offset: horizontal_pos[i] + offset_2 + 1,
                           vertical_pos[i] - offset: vertical_pos[i] + offset_2 + 1
                           ]

        tile_metadata = [
            (horizontal_pos[i] - offset, horizontal_pos[i] + offset_2 + 1),
            (vertical_pos[i] - offset, vertical_pos[i] + offset_2 + 1)]

        metadata.append(tile_metadata)
        payload.append(tile)

    payload_array = np.stack(payload)

    return [payload_array, metadata]


def extract_labels_and_cloud_tiles(swath_array, file_path, tile_size=3, stride=3):
    """
    :param swath_array: numpy of a swath
    :param file_path: filepath to original hdf - for contextualising errors
    :param tile_size: size of tile (default 3)
    :param stride: space between tiles (
    :return: nested list of [labelled payload, unlabelled payload, labelled meta, unlabelled meta]
    """

    if swath_array.shape != (27, 2030, 1350):
        raise ValueError("Tiles are extracted only from swaths with label mask")
        
    labelled_payload, labelled_metadata = extract_label_tiles(swath_array=swath_array,
                                                              file_path=file_path,
                                                              tile_size=tile_size)

    number_of_labels = len(labelled_payload)

    unlabelled_payload, unlabelled_metadata = extract_random_sample_where_clouds(
        swath_array=swath_array,
        file_path=file_path,
        number_of_labels=number_of_labels,
        tile_size=tile_size,
        stride=stride)

    return [labelled_payload, unlabelled_payload, labelled_metadata, unlabelled_metadata]


if __name__ == "__main__":

    mask = get_sampling_mask((3, 3), 3)
    assert np.sum(mask) == 1

    mask = get_sampling_mask((4, 4), 3)
    assert np.sum(mask) == 4

    mask = get_sampling_mask((4, 4), 4)
    assert np.sum(mask) == 1

    mask = get_sampling_mask((5, 5), 4)
    assert np.sum(mask) == 4

    labelmask = get_label_mask(np.ones((3, 5, 5)))
    assert np.all(labelmask == 1)
    assert labelmask.shape == (5, 5)

    classes = np.zeros((3, 5, 6))
    classes[0, 2:4, 1] = 3
    classes[1, 1, 1] = 1
    labelmask = get_label_mask(classes)

    assert np.sum(labelmask) == 3
    assert labelmask[2, 1] == 1
    assert labelmask[3, 1] == 1
    assert labelmask[1, 1] == 1
    assert labelmask.shape == (5, 6)
