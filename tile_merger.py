""" Set of tiles""" 

import glob
import numpy as np
import sys
import os

def load_dir_npys(dir_path):

    for dirs,_,_ in os.walk(dir_path):

        if "tiles" in dirs:
            for array in glob.glob(os.path.join(dirs, "*.npy")):

                yield np.load(array)

if __name__ == "__main__":

    root_dir = sys.argv[1]

    save_dir = os.path.join(root_dir, "all-tiles")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0
    for swath in load_dir_npys(root_dir):

        for tile in swath:
            
            np.save(os.path.join(save_dir, "{}.npy".format(i)), tile)
            i += 1