import numpy as np
import os
import random

from tqdm import tqdm

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

def get_label_mask(labels):
    """ given the class occurences channels over 10 layers, returns a 2d array marking as 1 the labelled pixels and as 0 the unlabelled ones."""

    label_mask = np.sum(~labels.mask, 3) > 0

    return label_mask 

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

def sample_cloudy_unlabelled_tiles(swath_tuple, cloud_mask, label_mask, number_of_tiles, tile_size=3):
    """
    :param swath_tuple: 
    :param cloud_mask: 2d array of zise (w, h) marking the cloudy pixels 
    :param label_mask: 2d array of zise (w, h) marking the labelled pixels 
    :param number_of_tiles: the number of tiles to sample. It is reset to maximal number of tiles that can be sampled, if bigger
    :param tile_size: size of the tile selected from within the image
    :return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles; and a list of tuples ((w1, w2), (h1, h2)) with the relative positions of the sampled tiles withing the swath
    The script will use a cloud_mask channel to mask away all non-cloudy data and a label_mask channel to mask away all labelled data. The script will then randomly select a number of tiles (:param number of tiles) from the cloudy areas that are unlabelled.
    """

    # mask out borders not to sample outside the swath
    allowed_pixels = get_sampling_mask((MAX_WIDTH, MAX_HEIGHT), tile_size)

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

    positions, tiles = [], [[] for _ in swath_tuple]
    for center in tile_centers:
        center_w, center_h = center

        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        tile_position = ((w1, w2), (h1, h2))

        positions.append(tile_position)   
        for i, a in enumerate(swath_tuple):     
            tiles[i].append(a[:, w1:w2, h1:h2])

    positions = np.stack(positions)
    for i, t in enumerate(tiles):     
        tiles[i] = np.stack(t)

    return tiles, positions


def extract_cloudy_labelled_tiles(swath_tuple, cloud_mask, label_mask, tile_size=3):
    """
    :param swath_tuple: input numpy array from MODIS of size (nb_channels, w, h)
    :param cloud_mask: 2d array of zise (w, h) marking the cloudy pixels 
    :param label_mask: 2d array of zise (w, h) marking the labelled pixels 
    :param tile_size: the size of the channels
    :return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles; and a list of tuples ((w1, w2), (h1, h2)) with the relative positions of the extracted tiles withing the swath
    The script will use a cloud_mask channel to mask away all non-cloudy data and a label_mask channel to mask away all unlabelled data. The script will then select all tiles from the cloudy areas that are labelled.
    """
    # mask not to sample outside the swath
    allowed_pixels = get_sampling_mask((MAX_WIDTH, MAX_HEIGHT), tile_size)

    # combine the three masks, tile centers will be sampled from the cloudy and labelled pixels that are not in the borders of the swath
    labelled_pixels = allowed_pixels & cloud_mask & label_mask

    assert labelled_pixels.sum() > 0, "Swath contains no valid labelled pixels."

    labelled_pixels_idx = np.where(labelled_pixels == 1)
    labelled_pixels_idx = list(zip(*labelled_pixels_idx))

    offset, offset_2 = get_tile_offsets(tile_size)

    positions, tiles = [], [[] for _ in swath_tuple]
    for center in labelled_pixels_idx:
        center_w, center_h = center

        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        tile_position = ((w1, w2), (h1, h2))

        positions.append(tile_position)   
        for i, a in enumerate(swath_tuple):     
            tiles[i].append(a[:, w1:w2, h1:h2])

    positions = np.stack(positions)
    for i, t in enumerate(tiles):     
        tiles[i] = np.stack(t)

    return tiles, positions

def sample_labelled_and_unlabelled_tiles(swath_tuple, cloud_mask, label_mask, tile_size=3):
    """
    :param swath_tuple: tuple of numpy arrays of size (C, H, W, ...) to be tiled coherently
    :param tile_size: size of tile (default 3)
    :param cloud_mask: mask where cloudy
    :param label_mask: mask where labels are available 
    :return: nested list of labelled tiles, unlabelled tiles, labelled tile positions, unlabelled tile positions
    Samples the same amount of labelled and unlabelled tiles from the cloudy data.
    """
        
    labelled_tiles, labelled_positions = extract_cloudy_labelled_tiles(swath_tuple, cloud_mask, label_mask, tile_size)

    number_of_labels = len(labelled_tiles[0])

    unlabelled_tiles, unlabelled_positions = sample_cloudy_unlabelled_tiles(swath_tuple, cloud_mask, label_mask, number_of_labels, tile_size)

    return labelled_tiles, unlabelled_tiles, labelled_positions, unlabelled_positions


if __name__ == "__main__":

    import glob
    import os

    from nc_loader import read_nc

    nc_dir = "/mnt/disks/cumulo-tiles/nc/"
    save_dir = "/mnt/disks/cumulo-tiles/npz/"

    for dr in [os.path.join(save_dir, "label"), os.path.join(save_dir, "unlabel")]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    file_paths = glob.glob(os.path.join(nc_dir, "*.nc"))

    if len(file_paths) == 0:
        raise FileNotFoundError("no nc files in", nc_dir)

    for filename in tqdm(file_paths):

        radiances, properties, cloud_mask, labels = read_nc(filename)
        label_mask = get_label_mask(labels)
        
        try:
            labelled_tiles, unlabelled_tiles, labelled_positions, unlabelled_positions = sample_labelled_and_unlabelled_tiles((radiances, properties, cloud_mask, labels), cloud_mask[0], label_mask[0])

            name = os.path.basename(filename).replace(".nc", "")

            save_name = os.path.join(save_dir, "label", name)
            np.savez_compressed(save_name, radiances=labelled_tiles[0].data, properties=labelled_tiles[1].data, cloud_mask=labelled_tiles[2].data, labels=labelled_tiles[3].data, location=labelled_positions)

            save_name = os.path.join(save_dir, "unlabel", name)
            np.savez_compressed(save_name, radiances=unlabelled_tiles[0].data, properties=unlabelled_tiles[1].data, cloud_mask=unlabelled_tiles[2].data, labels=unlabelled_tiles[3].data, location=unlabelled_positions)

        except AssertionError as e:
            print(filename, e)

