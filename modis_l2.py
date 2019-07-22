#!usr/bin/env python
from pyhdf.SD import SD, SDC
import pyhdf.HDF as HDF
import glob
import pickle
import datetime

def get_l2_filename(filename_example, root):
    #MYD021KM.A2008003.1855.061.2018031033116.hdf l1
    #MYD06_L2.A2008001.0000.061.2018031060235.hdf l2 
    parts = filename_example.split('.')
    l2_filename = glob.glob(root + 'MYD06_L2.{}.{}.*.hdf'.format(parts[1], parts[2]))[0]
    return l2_filename

def run(l1_filename, root_dir, output_dir = ''):
	#output_filename = filename.split('A')[1].split('.')
	#dt = datetime.datetime(year = int(output_filename[0][:4]), month = 1, day = 1, hour = int(output_filename[1][:2]), minute = int(output_filename[1][2:]) )
	#dt = dt + datetime.timedelta(days = int(output_filename[0][4:])-1)
	filename = get_l2_filename(l1_filename, root_dir)
	level_data = SD(filename, SDC.READ)
	latitude = level_data.select('Latitude').get()
	longitude = level_data.select('Longitude').get()
	latitude = latitude[:][:1350]
	longitude = longitude[:][:1350]
	lwp = level_data.select('Cloud_Water_Path').get()
	lwp = lwp[:1350]
	cod = level_data.select('Cloud_Optical_Thickness').get()
	cod = cod[:1350]
	cloud_mask = level_data.select('Cloud_Mask_1km').get()[:,:1350,0]
	#with open(output_dir + output_filename, 'w') as f:
	#	pickle.dump([lwp, cod, cloud_mask, latitude, longitude, dt], f)
	return lwp, cod, cloud_mask

