import utils
import sys
import numpy as np

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.rstrip()
        swath = np.load(line, allow_pickle=True)
        swath = utils.fill_all_channels(swath)
        print("{} interpolated".format(swath))
