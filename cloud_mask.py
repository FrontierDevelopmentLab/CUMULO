from satpy import Scene
import numpy as np
import glob
import os
import sys

def get_cloud_mask(cloud_mask_dir, level_1_filename):
    
    head, tail = os.path.split(level_1_filename)
    head_parts = head.split("/")

    cloud_mask_filename = glob.glob(os.path.join(cloud_mask_dir, head_parts[-3], head_parts[-2], head_parts[-1], '*' + level_1_filename.split('.A')[1][:12] + '*'))[0]

    swath = Scene(reader = 'modis_l2', filenames = [cloud_mask_filename, level_1_filename])
    swath.load(['cloud_mask'], resolution = 1000)

    cloud_mask = np.array(swath['cloud_mask'].load())[:, :1350]
    cloud_mask = cloud_mask > 0
    cloud_mask = cloud_mask.astype(int)
    
    return cloud_mask
