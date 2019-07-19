import utils
import sys
import numpy as np

if __name__ == "__main__":
    for line in sys.stdin:
        swath = np.load(line)
        swath = utils.fill_all_channels(swath)
        print("{} interpolated".format(swath))
