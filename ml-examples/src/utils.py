import netCDF4 as nc4
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.loader import read_npz

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_dataset_statistics(dataset, nb_classes, batch_size, collate, use_cuda=True, tile_size=9):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate)
    weights = np.zeros(nb_classes)
    sum_x = torch.zeros(13)
    std = torch.zeros(1, 13, 1, 1)

    if use_cuda:
        sum_x = sum_x.cuda()
        std = std.cuda()
    
    nb_tiles = 0
    for tiles, labels in dataloader:
        
        nb_tiles += len(tiles)

        if use_cuda:
            tiles = tiles.cuda()

        # class weights
        weights += np.histogram(labels, bins=range(nb_classes+1), normed=False)[0]
        
        sum_x += torch.sum(tiles, (0, 2, 3))
    
    nb_pixels = nb_tiles * tile_size
    m = (sum_x / nb_pixels).reshape(1, 13, 1, 1)
    
    for tiles, _ in dataloader:
        
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

def get_tile_sampler(dataset, allowed_idx=None, ext="npz"):

    indices = []
    paths = dataset.file_paths.copy()

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]
    
    for i, swath_path in enumerate(paths):

        swath, *_ = read_npz(swath_path)
        
        indices += [(i, j) for j in range(swath.shape[0])]

    return SubsetRandomSampler(indices)

def tile_collate(swath_tiles):
    
    data = np.vstack([tiles for _, tiles, _, _, _ in swath_tiles])
    target = np.hstack([labels for *_, labels in swath_tiles])

    return torch.from_numpy(data).float(), torch.from_numpy(target).long()
