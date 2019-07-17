#!usr/bin/env python
from pyhdf.SD import SD, SDC
import pyhdf.HDF as HDF
import glob
import pickle
import matplotlib.pyplot as plt

def run(path, output_dir = ''):
	for filename in glob.glob(path + '*MYD*.hdf'):
		output_filename = filename.split('A')[1].split('.')
		output_filename = '{}{}.pkl'.format(output_filename[0], output_filename[1])
		level_data = SD(filename, SDC.READ)
		latitude = level_data.select('Latitude').get()
		longitude = level_data.select('Longitude').get()
		latitude = latitude[:][:1350]
		longitude = longitude[:][:1350]
		lwp = level_data.select('Cloud_Water_Path').get()
		lwp = lwp[:1350]
		cod = level_data.select('Cloud_Optical_Thickness').get()
		cod = cod[:1350]
		cloud_mask = level_data.select('Cloud_Mask_1km').get()[:,:1350,:]
		with open(output_dir + output_filename, 'w') as f:
			pickle.dump([lwp, cod, cloud_mask, latitude, longitude], f)
		return
run('./temp/')
