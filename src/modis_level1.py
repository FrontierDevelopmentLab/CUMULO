import glob
import numpy as np
import os

from satpy import Scene

MAX_WIDTH, MAX_HEIGHT = 1354, 2030

def find_matching_geoloc_file(radiance_filename, myd03_dir):
    """
    :param radiance_filename: the filename for the radiance .hdf, demarcated with "MYD02".
    :param myd03_dir: root directory of MYD03 geolocational files
    :return geoloc_filename: the path to the corresponding geolocational file, demarcated with "MYD03"
    The radiance (MYD02) geolocational (MYD03) files share the same capture date (saved in the filename itself), yet can have different processing dates (also seen within the filename). A regex search on a partial match in the same directory provides the second filename and path.
    """

    tail = os.path.basename(radiance_filename)
    identifier = tail.split('A')[1].split('.')[1]
    geoloc_filename = glob.glob(os.path.join(myd03_dir, '*D03*.{}.*.hdf'.format(identifier)))[0]

    return geoloc_filename

def find_all_radiance_geoloc_pairs(path):
    """
    :param path: directory containing both radiance and geolocation files as .hdf, demarcated respectively with "MYD02" and "MYD03".
    :return pairs: a list of filename pairs (radiance_filename, geoloc_filename)
    The radiance (MYD02) geolocational (MYD03) files share the same capture date (saved in the filename itself), yet can have different processing dates (also seen within the filename).
    """

    radiance_filenames = glob.glob(path + '*D021KM*.hdf')

    pairs = []

    for radiance_filename in radiance_filenames:

        geoloc_filename = find_matching_geoloc_file(radiance_filename)
        pairs.append([radiance_filename, geoloc_filename])

    return pairs

def get_swath(radiance_filename, myd03_dir):
    """
    :param radiance_filename: MYD02 filename
    :param myd03_dir: root directory of MYD03 geolocational files
    :return swath: numpy.ndarray of size (15, HEIGHT, WIDTH) 
    Uses the satpy Scene reader with the modis-l1b files. Issues reading files might be due to pyhdf not being
    installed - otherwise try pip install satpy[modis_0l1b]
    Creates a scene with the MYD02 and MYD03 files, and extracts them as multi-channel arrays. The lat and long are
    are appended as additional channels.
    """

    # bands selected from MODIS
    composite = ['1', '2', '29', '33', '34', '35', '36', '26', '27', '20', '21', '22', '23']
    
    # find a corresponding geolocational (MYD03) file for the provided radiance (MYD02) file
    geoloc_filename = find_matching_geoloc_file(radiance_filename, myd03_dir)

    # load the global scene using satpy
    global_scene = Scene(reader='modis_l1b', filenames=[radiance_filename, geoloc_filename])

    # load composite, resolution of 1km
    global_scene.load(composite, resolution=1000)

    # load latitudes and longitudes, resolution 1km
    global_scene.load(['latitude', 'longitude'], resolution=1000)
    latitude = np.array(global_scene['latitude'].load())
    longitude = np.array(global_scene['longitude'].load())

    swath = []

    for comp in composite:
        temp = np.array(global_scene[comp].load())
        swath.append(temp[:MAX_HEIGHT, :MAX_WIDTH])

    swath.append(latitude[:MAX_HEIGHT, :MAX_WIDTH])
    swath.append(longitude[:MAX_HEIGHT, :MAX_WIDTH])

    return np.array(swath, dtype=np.float16)

def get_swath_rgb(radiance_filename, myd03_dir, composite='true_color'):
    """
    :param radiance_filename: MYD02 filename
    :param myd03_dir: root directory of MYD03 geolocational files
    :return visible RGB channels: numpy.ndarray of size (3, 2030, 1354) 
    Uses the satpy Scene reader with the modis-l1b files. Issues reading files might be due to pyhdf not being
    installed - otherwise try pip install satpy[modis_0l1b]
    Creates a scene with the MYD02 file, and extracts the RGB channels from the 1, 4, 3 visible MODIS bands.
    """

    # find a corresponding geolocational (MOD03) file for the provided radiance (MYD02) file
    geoloc_filename = find_matching_geoloc_file(radiance_filename, myd03_dir)

    global_scene = Scene(reader='modis_l1b', filenames=[radiance_filename, geoloc_filename])

    # load it in, make sure resolution is 1000 to match our other datasets
    global_scene.load([composite], resolution=1000)

    rgb = np.array(global_scene[composite])[:,:MAX_HEIGHT,:MAX_WIDTH]

    return rgb
