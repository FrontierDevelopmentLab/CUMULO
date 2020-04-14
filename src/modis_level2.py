import glob
import numpy as np
import os

from pyhdf.SD import SD, SDC
from satpy import Scene

'''take in the MODIS level 1 filename to get the information needed to find the corresponding MODIS level 2 filename.
This info includes the YYYY and day in year (ex: AYYYYDIY) and then the time of the pass (ex1855)
It returns the full level 2 filename path'''

MAX_WIDTH, MAX_HEIGHT = 1354, 2030

def get_matching_l2_filename(radiance_filename, l2_dir):
    """
    :param radiance_filename: the filename for the radiance .hdf, demarcated with "MOD021KM".
    :param l2_dir: the root directory containing the l2 files.
    :return l2_filename: the path to the corresponding l2 file, demarcated with "MYD06_L2"
    The radiance (MOD021KM) and level2 (MYD06_L2) files share the same capture date (saved in the ilename itself), yet can have different processing dates (also seen within the filename). A regex search on a partial match in the same directory provides the second filename and path.
    """

    # MYD021KM.A2008003.1855.061.2018031033116.hdf l1
    # MYD06_L2.A2008003.1855.061.2018031060235.hdf l2

    tail = os.path.basename(radiance_filename)
    tail_parts = tail.split('.')
    
    l2_filename = glob.glob(os.path.join(l2_dir, 'MYD06_L2.{}.{}.*.hdf'.format(tail_parts[1], tail_parts[2])))[0]
    return l2_filename

def get_channels(l1_filename, l2_dir):

    """ take in the level 1 filename and rootdir for the level 2 data, returns an np.array of size (10, HEIGHT, WIDTH) with all l2 channels"""

    filename = get_matching_l2_filename(l1_filename, l2_dir)

    level_data = SD(filename, SDC.READ)

    lwp = level_data.select('Cloud_Water_Path').get()[:MAX_HEIGHT,:MAX_WIDTH]
    cod = level_data.select('Cloud_Optical_Thickness').get()[:MAX_HEIGHT,:MAX_WIDTH]
    cer = level_data.select('Cloud_Effective_Radius').get()[:MAX_HEIGHT,:MAX_WIDTH]
    cpop = level_data.select('Cloud_Phase_Optical_Properties').get()[:MAX_HEIGHT,:MAX_WIDTH]

    ctp = level_data.select('cloud_top_pressure_1km').get()[:MAX_HEIGHT,:MAX_WIDTH]
    cth = level_data.select('cloud_top_height_1km').get()[:MAX_HEIGHT,:MAX_WIDTH]
    ctt = level_data.select('cloud_top_temperature_1km').get()[:MAX_HEIGHT,:MAX_WIDTH]
    cee = level_data.select('cloud_emissivity_1km').get()[:MAX_HEIGHT,:MAX_WIDTH]

    st = level_data.select('surface_temperature_1km').get()[:MAX_HEIGHT,:MAX_WIDTH]
    
    channels = np.stack([lwp, cod, cer, cpop, ctp, cth, ctt, cee, st])

    return channels.astype(np.float16)

def get_cloud_mask(l1_filename, cloud_mask_dir):
    
    """ return a 2d mask, with cloudy pixels marked as 1, non-cloudy pixels marked as 0 """
    
    basename = os.path.split(l1_filename)

    cloud_mask_filename = glob.glob(os.path.join(cloud_mask_dir, 'MYD35*' + l1_filename.split('.A')[1][:12] + '*'))[0]
    
    # satpy returns(0=Cloudy, 1=Uncertain, 2=Probably Clear, 3=Confident Clear)
    swath = Scene(reader = 'modis_l2', filenames = [cloud_mask_filename])
    swath.load(['cloud_mask'], resolution = 1000)
    
    cloud_mask = np.array(swath['cloud_mask'].load())[:MAX_HEIGHT, :MAX_WIDTH]

    cloud_mask = (cloud_mask == 0)
    cloud_mask = cloud_mask.astype(np.intp)
    
    return cloud_mask

if __name__ == "__main__":
    
    l1_path = sys.argv[1]
    cloudmask_dir = "../DATA/aqua-data/cloud_mask/"

    save_dir = "../DATA/aqua-data-processed/cloud_mask/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cloudmask = get_cloud_mask(l1_path, cloudmask_dir)

    np.save(os.path.join(save_dir, os.path.basename(l1_path).replace(".hdf", ".npy")), cloudmask)
