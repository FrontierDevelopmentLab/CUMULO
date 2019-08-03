from pyhdf.SD import SD, SDC
import pyhdf.HDF as HDF
import numpy as np
import glob
import datetime
import sys
import cloudsat_tools
import time as t
from satpy import Scene
import matplotlib.pyplot as plt
from deco import synchronized, concurrent

modis_aux_dir = '/mnt/disks/sdb/modis_aux/'
cloudsat_classes = '/mnt/disks/sdb/cloudsat_CC/cloudsat_CC/'
modis_l1_dir = '/mnt/disks/sdb/l1_aqua/2008/01/01/'
couldntdo = []

def get_date(filename):
    date = filename.split('/')[-1].split('_')[0]
    date = datetime.datetime(year = 2008, month = 1, day = 1, hour = int(date[7:9]), minute = int(date[9:11])) + datetime.timedelta(days = int(date[4:7]) -1)
    return date


def nearest_modis_time(date):
        minute = int(date.minute)
        closest = (5 * round(minute/5))
        if closest < minute:
                closest += 5
        if (closest%5) != 0:
                raise ValueError('Minute in nearest modis time is not divisible by 5')
        if closest > 59:
                hour = date.hour
                date = date + datetime.timedelta(seconds = (60 * (closest - minute)))
                print(date, 'went over the hour')
        else:
            date = date.replace(minute = int(closest))
        print(date, minute, 'new date and old minute')
        if date.minute%5 != 0 :
            print('You messed up rolling over the hour')
            raise Exception
        return date

#@concurrent
def save_collocated_cloudsat(filename):
	diy = filename.split('2008')[1][:3]
	identifier = filename.split('/')[-1].split('_')[0]
	date = get_date(filename)
	aux_hdf = SD(filename, SDC.READ)
	#from http://www.cloudsat.cira.colostate.edu/sites/default/files/products/files/MOD06-AUX_PDICD.P1_R05.rev0_.pdf
	#we only take the 8th pixel because thats where the footprints overlap
	#modis along track index
	mxti = aux_hdf.select('MODIS_pixel_index_across_track').get()[:, 8]    
	#modis cross track index
	mati = aux_hdf.select('MODIS_pixel_index_along_track').get()[:, 8]
	modis_lat = aux_hdf.select('MODIS_latitude').get()[:, 8]
	modis_lon = aux_hdf.select('MODIS_longitude').get()[:, 8]
	#time in seconds
	aux_data = HDF.HDF(filename)
	time = cloudsat_tools.get_1D_var(aux_data, 'Profile_time')
	#load in the cloud classes for this granule
	cloudsat_filename = glob.glob(cloudsat_classes + identifier + '*.hdf')[0]
	cloudsat_sd = SD(cloudsat_filename, SDC.READ)
	classes = cloudsat_sd.select('CloudLayerType').get()
	start_points = []
	end_points = []
	#get the start and end points by finding where the index for mati is 1
	for i in range(len(mati)):
		if mati[i] == 1:
			if (i not in start_points) and ((i - 1) not in start_points) and ((i-2) not in start_points):
				if len(start_points) >= 1:
					end_points.append(i-1)
				start_points.append(i)
	#the final end point will be the end
	end_points.append(-1)
	for start, end in zip(start_points, end_points):
		#modis time
		modis_time = nearest_modis_time(date + datetime.timedelta(seconds = int(time[start][0] - 60)))
		print(modis_time.hour, modis_time.minute,'modis time')
		t.sleep(10)
		#format output
		format_output_string = '.A2008{}.{}'.format(diy, modis_time.strftime('%H%M'))
		test_filename = glob.glob(modis_l1_dir + '*' + format_output_string + '*.hdf')
		print(test_filename)
        #only run if we have 2 filenames becuase we need both MYD021 and MYD03 to run Scene()
		if len(test_filename) == 2:
			swath = Scene(reader = 'modis_l1b', filenames = test_filename)
			swath.load(['latitude', 'longitude'], resolution = 1000)
			lat = np.array(swath['latitude'].load())
			lon = np.array(swath['longitude'].load())
			#technically modis-aux can go up to 2040 and 1354 since these are acceptable sizes of modis swaths. cut down at the end to 2030 x 1350
			false_array = np.zeros((2040, 1354, 8))
			#for each lat index in mati find the lon index of the pixel and place there
			for lat_index, cc, pixel_lon, pixel_lat in zip(mati[start:end], classes[start:end], modis_lon[start:end], modis_lat[start:end]):
				#lat index of mati starts at 1 so you need to -1 to make 0 index
				lat_index = lat_index - 1
				search_along = lon[lat_index][800:900]
				lon_index = min([(abs(l-pixel_lon), n) for n, l in enumerate(search_along)])[1] + 800
				counts = [0 for count_n in range(8)]
				print(lat[lat_index][lon_index], pixel_lat)
				cc = cc.tolist()
				for c in range(8):
					counts[c] = cc.count(c + 1)
				false_array[lat_index][lon_index] = counts
			false_array = false_array[:2030, :1350, :]
			output_name = 'CC' + format_output_string + '.npy'
			np.save('temp/'+ output_name, false_array)	
	return
	
def trycatch(modaux):
	try:
		save_collocated_cloudsat(modaux)
	except IndexError:
		print('you suck couldnt do', modaux)
		couldntdo.append(modaux)

#@synchronized
def run(filenames):
	for modis_aux_filename in filenames:
		print(modis_aux_filename)
		trycatch(modis_aux_filename)
	return


run_filenames = sorted(glob.glob(modis_aux_dir + '2008001*.hdf'))

run(run_filenames[10:11])
#print(couldntdo, 'couldnt do you suck')
