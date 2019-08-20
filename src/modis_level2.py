import glob
import numpy as np
import os

from pyhdf.SD import SD, SDC
from satpy import Scene

'''take in the MODIS level 1 filename to get the information needed to find the corresponding MODIS level 2 filename.
This info includes the YYYY and day in year (ex: AYYYYDIY) and then the time of the pass (ex1855)
It returns the full level 2 filename path'''

def get_matching_l2_filename(radiance_filename, l2_dir):
    """
    :param radiance_filename: the filename for the radiance .hdf, demarcated with "MOD021KM".
    :param l2_dir: the root directory containing the l2 files.
    :return l2_filename: the path to the corresponding l2 file, demarcated with "MYD06_L2"
    The radiance (MOD021KM) and geolocational (MYD06_L2) files share the same capture date (saved in the ilename itself), yet can have different processing dates (also seen within the filename). A regex search on a partial match in the same directory provides the second filename and path.
    """

    # MYD021KM.A2008003.1855.061.2018031033116.hdf l1
    # MYD06_L2.A2008003.1855.061.2018031060235.hdf l2

    head, tail = os.path.split(radiance_filename)
    tail_parts = tail.split('.')
    head_parts = head.split('/')
    
    l2_filename = glob.glob(os.path.join(l2_dir, head_parts[-3], head_parts[-2], head_parts[-1], 'MYD06_L2.{}.{}.*.hdf'.format(tail_parts[1], tail_parts[2])))[0]
    return l2_filename

def get_lwp_cod_ctp(l1_filename, l2_dir):

    """ take in the level 1 filename and rootdir for the level 2 data, returns liquid water path (lwp), cloud top pressure (ctp) and cloud optical depth (cod)"""

    filename = get_matching_l2_filename(l1_filename, l2_dir)

    level_data = SD(filename, SDC.READ)

    lwp = level_data.select('Cloud_Water_Path').get()[:2030,:1350].tolist()
    cod = level_data.select('Cloud_Optical_Thickness').get()[:2030,:1350].tolist()
    ctp = level_data.select('cloud_top_pressure_1km').get()[:2030,:1350].tolist()
    
    channels = np.stack([lwp, cod, ctp])

    return channels.astype(np.float16)

def get_cloud_mask(l1_filename, cloud_mask_dir):
    
    """ return a 2d mask, with cloudy pixels marked as 1, non-cloudy pixels marked as 0 """
    
    head, tail = os.path.split(l1_filename)
    head_parts = head.split("/")

    cloud_mask_filename = glob.glob(os.path.join(cloud_mask_dir, head_parts[-3], head_parts[-2], head_parts[-1], '*' + l1_filename.split('.A')[1][:12] + '*'))[0]
    
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

    save_dir = "../DATA/aqua-data-processed/cloud_mask/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cloudmask = get_cloud_mask(l1_path, cloudmask_dir)

    np.save(os.path.join(save_dir, os.path.basename(l1_path).replace(".hdf", ".npy")), cloudmask)
