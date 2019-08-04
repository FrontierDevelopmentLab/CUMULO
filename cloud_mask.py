from satpy import Scene
import numpy as np
import glob
import os
import sys

def get_cloud_mask(level_1_filename, cloud_mask_dir):
    
    """ return a 2d mask, with cloudy pixels marked as 1, non-cloudy pixels marked as 0 """
    
    head, tail = os.path.split(level_1_filename)
    head_parts = head.split("/")

    cloud_mask_filename = glob.glob(os.path.join(cloud_mask_dir, head_parts[-3], head_parts[-2], head_parts[-1], '*' + level_1_filename.split('.A')[1][:12] + '*'))[0]
    
    # satpy returns(0=Cloudy, 1=Uncertain, 2=Probably Clear, 3=Confident Clear)
    swath = Scene(reader = 'modis_l2', filenames = [cloud_mask_filename])
    swath.load(['cloud_mask'], resolution = 1000)
    
    cloud_mask = np.array(swath['cloud_mask'].load())[:2030, :1350]
    cloud_mask = (cloud_mask == 0)
    cloud_mask = cloud_mask.astype(int)
    
    return cloud_mask

if __name__ == "__main__":
    
    l1_path = sys.argv[1]
    cloudmask_dir = "../DATA/aqua-data/cloud_mask/"

    save_dir = "../DATA/aqua-data-processed/cloudmask/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cloudmask = get_cloud_mask(l1_path, cloudmask_dir)

    np.save(os.path.join(save_dir, os.path.basename(l1_path).replace(".hdf", ".npy")), cloudmask)

