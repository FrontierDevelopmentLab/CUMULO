#!usr/bin/env python
from satpy import Scene, available_readers
from satpy.writers import to_image, get_enhanced_image
from PIL import Image, ImageDraw
import glob
import numpy as np
import sys
from netCDF4 import Dataset
import image_slicer

new_image_size = (2200, 1320)

#split the array according to some preset conditions from duncan WP
def split_array(arr):
    v_splits = np.array_split(arr, [440, 880, 1320, 1760], axis=0)
    return sum((np.array_split(v_split, [440, 880], axis=1) for v_split in v_splits), [])

#get the filenames for the radiances and the geolocation information
def get_modis_filenames(path):
	#first find the radiance filename
	radiance_filename = glob.glob(path + '*MYD021KM*.hdf')[0]
	geoloc_filename = glob.glob(path + '*MYD03*')[0]
	
	#returns in the expected format, geoloc then radiance
	return [radiance_filename, geoloc_filename]
	
#correct for the bowtie effect on the edges of the image
def bowtie_correction(field, corr_param, path = ''):
	correction_filename = path + 'bowtie_correction_1km_{}'.format(corr_param) + '.nc'
	dataset = Dataset(correction_filename, mode = 'r')
	along_track_index = dataset['at_ind'][:].astype('int')
	cross_track_index = dataset['ct_ind'][:].astype('int')
	field_1km = field[:, :, :1350]
	field_1km.values = field_1km.values[:, along_track_index, cross_track_index]
	return field_1km

def run(path, composite = 'day_microphysics', tiles = None, save_dir = './test_dump/'):
	#given a path, find all the two correct files within
	files = get_modis_filenames(path)
	#load the global scene using satpy
	#expects filenames to be radiance then geoloc
	#filenames must be in specific format, chop off beginning to work
	global_scene = Scene(reader = 'modis_l1b', filenames = files)
	#for future reference here are all available datasets
	available_datasets = global_scene.available_dataset_names()
	#want to create a cloudtop composite
	#for future reference all available composites
	available_composites = global_scene.available_composite_names()
	#print available_composites
	#expects the composite to be in a list
	global_scene.load([composite], resolution = 1000)
	#if you want a pic to pop up of the composite uncomment below
	#global_scene.show(composite)
	
	#correction of the bowtie
	da = global_scene[composite].load()
	correction = da.area.shape[0]
	da = bowtie_correction(da, correction)
	
	#pil image
	pil_image = get_enhanced_image(da).pil_image()
	#check if the image is not the correct shape
	if (pil_image.size[1] in [2030, 2040]) and (pil_image.size[0] in [1354, 1350]):
		image_name = 'fixthis.png'
		pil_image.save(image_name)
		if tiles:
			tiles = image_slicer.slice('fixthis.png', tiles, save = False)
			for i, tile in enumerate(tiles):
				tile.save(save_dir + '{}.png'.format(i))
			return image_name
	return image_name
