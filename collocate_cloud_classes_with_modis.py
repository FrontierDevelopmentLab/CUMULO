from pyhdf.SD import SD, SDC
import pyhdf.HDF as HDF
import numpy as np
import glob
import datetime
import sys
import cloudsat_tools
from satpy import Scene
from deco import synchronized, concurrent


def get_date(filename):
    date = filename.split('/')[-1].split('_')[0]
    date = datetime.datetime(year = 2008, month = 1, day = 1, hour = int(date[7:9]), minute = int(date[9:11])) + datetime.timedelta(days = int(date[4:7]) -1)
    return date

def nearest_modis_time(date):
    minute = date.minute
    closest = ((minute / 5) * 5)
    date = date.replace(minute = closest)
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
    for i in range(len(mati)):
        if mati[i] == 1:
            if (i not in start_points) and ((i - 1) not in start_points) and ((i-2) not in start_points):
                if len(start_points) >= 1:
                    end_points.append(i-1)
                start_points.append(i)

    for start, end in zip(start_points, end_points):
        #modis time
        modis_time = nearest_modis_time(date + datetime.timedelta(seconds = int(time[start][0])))
        #format output
        format_output_string = '.A2008{}.{}'.format(diy, modis_time.strftime('%H%M'))
        test_filename = glob.glob(modis_l1_dir + '*' + format_output_string + '*.hdf')
        swath = Scene(reader = 'modis_l1b', filenames = test_filename)
        swath.load(['latitude', 'longitude'], resolution = 1000)
        lat = np.array(swath['latitude'].load())[:, :1350]
        lon = np.array(swath['longitude'].load())[:, :1350]
        
        #technically modis-aux can go up to 2040 and 1354 since these are acceptable sizes of modis swaths. cut down at the end to 2030 x 1350
        false_array = np.zeros((2040, 1354))
      
        for lat_index, cc, pixel_lon in zip(mati[start:end], classes[start:end], modis_lon[start:end]):
            lat_index = lat_index - 1
            search_along = lon[lat_index][870:880]
            lon_index = min([(abs(l-pixel_lon), n) for n, l in enumerate(search_along)])[1] + 870
            cc = list(set([c for c in cc if c != 0]))
            if len(cc) > 1:
                cc = [9]
            if len(cc) == 0:
                cc = [0]
            false_array[lat_index][lon_index] = cc[0]
        false_array = false_array[:2030, :1350]
        output_name = 'CC' + format_output_string + '.npy'
        np.save('/home/rosealyd/level_2/collocated_classes/' + output_name, false_array)
    return

#@synchronized
def run(filenames):
    for modis_aux_filename in filenames:
        print modis_aux_filename
        save_collocated_cloudsat(modis_aux_filename)
    return

modis_aux_dir = '/home/rosealyd/aux/'
cloudsat_classes = '/home/rosealyd/cloudsat_classes/'
modis_l1_dir = '/home/rosealyd/level_1/2008/01/01/'
filenames = sorted(glob.glob(modis_aux_dir + '2008002*.hdf'))

run(filenames)