from extract_payload import extract_labels_and_cloud_tiles
import numpy as np

file_name = "tester_string_file"
input_swath = np.load(
    "../cloud_fxns/results_aqua_semisuper-sequential_swath_daylight_swath_MYD021KM.A2008001.0100.061.2018031001704.npy")

a, b, c, d = extract_labels_and_cloud_tiles(input_swath, file_name)