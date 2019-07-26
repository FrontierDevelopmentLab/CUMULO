from satpy import Scene
import numpy as np
import glob
import sys

def get_cloud_mask(cloud_mask_dir, level_1_filename):
    cloud_mask_filename = glob.glob(cloud_mask_dir + '*' + level_1_filename.split('.A')[1][:12] + '*')[0]
    print cloud_mask_filename
    swath = Scene(reader = 'modis_l2', filenames = [cloud_mask_filename, level_1_filename])
    swath.load(['cloud_mask'], resolution = 1000)
    cloud_mask = np.array(swath['cloud_mask'].load())[:, :1350]
    cloud_mask = cloud_mask > 0
    cloud_mask = cloud_mask.astype(int)
    return cloud_mask
