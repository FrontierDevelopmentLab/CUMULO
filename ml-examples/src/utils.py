import glob
import netCDF4 as nc4
import numpy as np
import os
import pickle

from skimage import io, transform
from skimage.io import imread
from skimage.util import crop

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_dataset_statistics(dataset, nb_classes, use_cuda=True):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8)
    weights = np.zeros(nb_classes)
    sum_x = torch.zeros(13)
    std = torch.zeros(1, 13, 1, 1)

    if use_cuda:
        sum_x = sum_x.cuda()
        std = std.cuda()
    
    nb_tiles = 0
    for x in dataloader:

        labels, tiles = x["labels"], x["tiles"].float()
        
        nb_tiles += len(tiles)

        if use_cuda:
            tiles = tiles.cuda()

        # class weights
        weights += np.histogram(labels, bins=range(nb_classes+1), normed=False)[0]
        
        sum_x += torch.sum(tiles, (0, 2, 3))
    
    nb_pixels = nb_tiles * 9
    m = (sum_x / nb_pixels).reshape(1, 13, 1, 1)
    
    for x in dataloader:
        
        tiles = x["tiles"].float()

        if use_cuda:
            tiles = tiles.cuda()

        std += torch.sum((tiles - m).pow(2), (0, 2, 3), keepdim=True)

    s = ((std / nb_pixels)**0.5)
    
    weights /= np.sum(weights)
    weights = 1 / (np.log(1.02 + weights))

    if use_cuda:
        m = m.cpu()
        s = s.cpu()

    return weights / np.sum(weights), m.reshape(13, 1, 1).numpy(), s.reshape(13, 1, 1).numpy()

class Normalizer(object):

    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, image):

        return (image - self.mean) / self.std

class TileExtractor(object):

    def __init__(self, t_width=3, t_height=3):

        self.t_width = t_width
        self.t_height = t_height

    def __call__(self, image):

        img_width = image.shape[1]
        img_height = image.shape[2]

        nb_tiles_row = img_width // self.t_width
        nb_tiles_col = img_height // self.t_height

        tiles = []
        locations = []

        for i in range(nb_tiles_row):
            for j in range(nb_tiles_col):

                tile = image[:, i * self.t_width: (i+1) * self.t_width, j * self.t_height: (j+1) * self.t_height]

                tiles.append(tile)

                locations.append(((i * self.t_width, (i+1) * self.t_width), (j * self.t_height, (j+1) * self.t_height)))

        tiles = np.stack(tiles)
        locations = np.stack(locations)

        return tiles, locations

# ------------------------------------------------------------ CUMULO HELPERS

def get_tile_sampler(dataset, allowed_idx=None):

    indices = []
    paths = dataset.swath_paths.copy()

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]
    
    for i, swath_name in enumerate(paths):

        swath_path = os.path.join(dataset.root_dir, swath_name)
        swath = np.load(swath_path)
        
        indices += [(i, j) for j in range(swath.shape[0])]

    return SubsetRandomSampler(indices)

def tile_collate(swath_tiles):
    
    data = np.vstack([s["tiles"] for s in swath_tiles])
    target = np.hstack([s["labels"] for s in swath_tiles])

    return {"tiles": torch.from_numpy(data), "labels": torch.from_numpy(target)}