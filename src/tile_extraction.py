import numpy as np
import os
import random

MAX_WIDTH, MAX_HEIGHT = 1354, 2030

# -------------------------------------------------------------------------------------------------- UTILS

def get_tile_offsets(tile_size):

    offset = tile_size // 2
    offset_2 = offset

    if not tile_size % 2:
        offset_2 -= 1

    return offset, offset_2

def get_sampling_mask(mask_shape=(MAX_HEIGHT, MAX_WIDTH), tile_size=3):
    """ returns a mask of allowed centers for the tiles to be sampled. The center of an even size tile is considered to be the point at the position (size // 2 + 1, size // 2 + 1) within the tile.
    """
    mask = np.ones(mask_shape, dtype=np.uint8)

    offset, offset_2 = get_tile_offsets(tile_size)

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

    return labelmask.astype(np.bool) 

def get_unlabel_mask(label_mask, tile_size=3):
    """returns inverse of label mask, with all pixels around a labelled one eroded."""

    offset, offset_2 = get_tile_offsets(tile_size)

    unlabel_mask = (~label_mask).copy()

    labelled_idx = np.where(label_mask)

    for center_w, center_h in zip(*labelled_idx):

        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        unlabel_mask[w1:w2, h1:h2] = False

    return unlabel_mask


# -------------------------------------------------------------------------------------------------- SAMPLERS

def sample_cloudy_unlabelled_tiles(swath_array, cloud_mask, label_mask, number_of_tiles, tile_size=3):
    """
    :param swath_array: input numpy array from MODIS of size (nb_channels, w, h)
    :param cloud_mask: 2d array of zise (w, h) marking the cloudy pixels 
    :param label_mask: 2d array of zise (w, h) marking the labelled pixels 
    :param number_of_tiles: the number of tiles to sample. It is reset to maximal number of tiles that can be sampled, if bigger
    :param tile_size: size of the tile selected from within the image
    :return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles; and a list of tuples ((w1, w2), (h1, h2)) with the relative positions of the sampled tiles withing the swath
    The script will use a cloud_mask channel to mask away all non-cloudy data and a label_mask channel to mask away all labelled data. The script will then randomly select a number of tiles (:param number of tiles) from the cloudy areas that are unlabelled.
    """

    # mask out borders not to sample outside the swath
    allowed_pixels = get_sampling_mask(swath_array.shape[1:], tile_size)

    # mask out labelled pixels and pixels around them
    unlabel_mask = get_unlabel_mask(label_mask)

    # combine the three masks, tile centers will be sampled from the cloudy and unlabelled pixels that are not in the borders of the swath
    unlabelled_pixels = np.logical_and.reduce([allowed_pixels, cloud_mask, unlabel_mask])
    unlabelled_pixels_idx = np.where(unlabelled_pixels == 1)
    unlabelled_pixels_idx = list(zip(*unlabelled_pixels_idx))

    number_of_tiles = min(number_of_tiles, len(unlabelled_pixels_idx))

    # sample without replacement
    tile_centers_idx = np.random.choice(np.arange(len(unlabelled_pixels_idx)), number_of_tiles, False)
    unlabelled_pixels_idx = np.array(unlabelled_pixels_idx)
    tile_centers = unlabelled_pixels_idx[tile_centers_idx]
    
    # compute distances from tile center of tile upper left and lower right corners
    offset, offset_2 = get_tile_offsets(tile_size)

    positions, tiles = [], []
    for center in tile_centers:
        center_w, center_h = center

        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        tile = swath_array[:, w1:w2, h1:h2]
        tile_position = ((w1, w2), (h1, h2))

        positions.append(tile_position)        
        tiles.append(tile)

    tiles = np.stack(tiles)
    positions = np.stack(positions)

    return tiles, positions


def extract_cloudy_labelled_tiles(swath_array, cloud_mask, label_mask, tile_size=3):
    """
    :param swath_array: input numpy array from MODIS of size (nb_channels, w, h)
    :param cloud_mask: 2d array of zise (w, h) marking the cloudy pixels 
    :param label_mask: 2d array of zise (w, h) marking the labelled pixels 
    :param tile_size: the size of the channels
    :return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles; and a list of tuples ((w1, w2), (h1, h2)) with the relative positions of the extracted tiles withing the swath
    The script will use a cloud_mask channel to mask away all non-cloudy data and a label_mask channel to mask away all unlabelled data. The script will then select all tiles from the cloudy areas that are labelled.
    """

    # mask not to sample outside the swath
    allowed_pixels = get_sampling_mask(swath_array.shape[1:], tile_size)

    # combine the three masks, tile centers will be sampled from the cloudy and labelled pixels that are not in the borders of the swath
    labelled_pixels = np.logical_and.reduce([allowed_pixels, cloud_mask, label_mask])
    labelled_pixels_idx = np.where(labelled_pixels == 1)
    labelled_pixels_idx = list(zip(*labelled_pixels_idx))

    offset, offset_2 = get_tile_offsets(tile_size)

    positions, tiles = [], []
    for center in labelled_pixels_idx:
        center_w, center_h = center

        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        tile = swath_array[:, w1:w2, h1:h2]
        tile_position = ((w1, w2), (h1, h2))

        positions.append(tile_position)        
        tiles.append(tile)

    tiles = np.stack(tiles)
    positions = np.stack(positions)

    return tiles, positions

def sample_labelled_and_unlabelled_tiles(swath_array, cloudmask_idx=24, labelmask_idx=(25, 26, 27, 28, 29, 30, 31, 32), tile_size=3):
    """
    :param swath_array: numpy of a swath
    :param tile_size: size of tile (default 3)
    :param cloudmask_idx: index of channel of swath_array corresponding to its cloud mask
    :param labelmask_idx: indices of channels of swath_array corresponding to the class occurences
    :return: nested list of labelled tiles, unlabelled tiles, labelled tile positions, unlabelled tile positions
    Samples the same amount of labelled and unlabelled tiles from the cloudy data.
    """

    cloud_mask = swath_array[cloudmask_idx]
    label_mask = get_label_mask(swath_array[np.array(labelmask_idx)])
        
    labelled_tiles, labelled_positions = extract_cloudy_labelled_tiles(swath_array, cloud_mask, label_mask, tile_size)

    number_of_labels = len(labelled_tiles)

    unlabelled_tiles, unlabelled_positions = sample_cloudy_unlabelled_tiles(swath_array, cloud_mask, label_mask, number_of_labels, tile_size)

    return labelled_tiles, unlabelled_tiles, labelled_positions, unlabelled_positions


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


    swath = np.ones((27, 5, 5))
    swath[18, :, 2] = 0
    swath[19:, :, :3] = 0

    labelled_tiles, unlabelled_tiles, labelled_positions, unlabelled_positions = sample_labelled_and_unlabelled_tiles(swath)

    assert labelled_tiles.shape == (3, 27, 3, 3)
    assert unlabelled_tiles.shape == labelled_tiles.shape
    assert labelled_positions.shape == (3, 2, 2)
    assert labelled_positions.shape == unlabelled_positions.shape

    assert np.all(labelled_positions[0] == ((0, 3), (2, 5)))
    assert np.all(labelled_positions[1] == ((1, 4), (2, 5)))
    assert np.all(labelled_positions[2] == ((2, 5), (2, 5)))

    assert np.all(unlabelled_positions[0][1] == (0, 3))
    assert np.all(unlabelled_positions[1][1] == (0, 3))
    assert np.all(unlabelled_positions[2][1] == (0, 3))
