#!usr/bin/env python
from pyhdf.SD import SD, SDC
import pyhdf.HDF as HDF
import glob
import numpy as np


'''take in the MODIS level 1 filename to get the information needed to find the corresponding MODIS level 2 filename.
This info includes the YYYY and day in year (ex: AYYYYDIY) and then the time of the pass (ex1855)
It returns the full level 2 filename path'''

def get_matching_l2_filename(radiance_filename):
    """
    :param radiance_filename: the filename for the radiance .hdf, demarcated with "MOD021KM".
    :return l2_filename: the path to the corresponding l2 file, demarcated with "MYD06_L2"
    The radiance (MOD021KM) and geolocational (MYD06_L2) files share the same capture date (saved in the ilename itself), yet can have different processing dates (also seen within the filename). A regex search on a partial match in the same directory provides the second filename and path.
    """

    # MYD021KM.A2008003.1855.061.2018031033116.hdf l1
    # MYD06_L2.A2008001.1855.061.2018031060235.hdf l2

    head, tail = os.path.split(radiance_filename)
    parts = tail.split('.')
    l2_filename = glob.glob(os.path.join(head, 'MYD06_L2.{}.{}.*.hdf'.format(parts[1], parts[2])))[0]
    return l2_filename

'''take in the level 1 filename and rootdir for the level 2 data, finds the cloud optical depth
(cod), liquid water path (lwp), and cloud mask for the swath'''


def run(l1_filename):
    filename = get_matching_l2_filename(l1_filename)
    level_data = SD(filename, SDC.READ)

    lwp = level_data.select('Cloud_Water_Path').get()[:,:1350].tolist()
    cod = level_data.select('Cloud_Optical_Thickness').get()[:,:1350].tolist()
    ctp = level_data.select('cloud_top_pressure_1km').get()[:,:1350].tolist()
    cth = level_data.select('Cloud_Top_Height').get()[:,:1350].tolist()
    
    #remove fill value of -9999 and replace with np.nan
    for i in range(len(lwp)):
        for j in range(len(lwp[i])):
            if lwp[i][j] == -9999:
                cod[i][j] = float('NaN')
                lwp[i][j] = float('NaN')
                ctp[i][j] = float('NaN')
                cth[i][j] = float('NaN')

    return np.array(lwp), np.array(cod), np.array(ctp), np.array(cth)
