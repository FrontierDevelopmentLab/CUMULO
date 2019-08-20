import glob
import numpy as np
import os

from satpy import Scene

def find_matching_geoloc_file(radiance_filename):
	"""
	:param radiance_filename: the filename for the radiance .hdf, demarcated with "MOD02".
	:return geoloc_filename: the path to the corresponding geolocational file, demarcated with "MOD03"
	The radiance (MOD02) geolocational (MOD03) files share the same capture date (saved in the filename itself), yet can have different processing dates (also seen within the filename). A regex search on a partial match in the same directory provides the second filename and path.
	"""

	head, tail = os.path.split(radiance_filename)
	identifier = tail.split('A')[1].split('.')[1]
	geoloc_filename = glob.glob(os.path.join(head, '*D03*.{}.*.hdf'.format(identifier)))[0]

	return geoloc_filename

def find_all_radiance_geoloc_pairs(path):
	"""
	:param path: directory containing both radiance and geolocation files as .hdf, demarcated respectively with "MOD02" and "MOD03".
	:return pairs: a list of filename pairs (radiance_filename, geoloc_filename)
	The radiance (MOD02) geolocational (MOD03) files share the same capture date (saved in the filename itself), yet can have different processing dates (also seen within the filename).
	"""

	radiance_filenames = glob.glob(path + '*D021KM*.hdf')

	pairs = []

	for radiance_filename in radiance_filenames:

		geoloc_filename = find_matching_geoloc_file(radiance_filename)
		pairs.append([radiance_filename, geoloc_filename])

	return pairs

def get_swath(radiance_filename):
	"""
	:param radiance_filename: MOD02 filename
	:return swath: numpy.ndarray of size (15, 2030, 1350) 
	Uses the satpy Scene reader with the modis-l1b files. Issues reading files might be due to pyhdf not being
	installed - otherwise try pip install satpy[modis_0l1b]
	Creates a scene with the MOD02 and MOD03 files, and extracts them as multi-channel arrays. The lat and long are
	are appended as additional channels.
	"""

	# bands selected from MODIS
	composite = ['1', '2', '29', '33', '34', '35', '36', '26', '27', '20', '21', '22', '23']
	
	# find a corresponding geolocational (MOD03) file for the provided radiance (MOD02) file
    geoloc_filename = find_matching_geoloc_file(radiance_filename)

	# load the global scene using satpy
	global_scene = Scene(reader='modis_l1b', filenames=[radiance_filename, geoloc_filename])

	# load composite, resolution of 1km
	global_scene.load(composite, resolution=1000)

	# load latitudes and longitudes, resolution 1km
	global_scene.load(['latitude', 'longitude'], resolution=1000)
	latitude = np.array(global_scene['latitude'].load())
	longitude = np.array(global_scene['longitude'].load())

	swath = []
	# note that we only take til 2030 and til 1350 to avoid any bowtie effects
	for comp in composite:
		temp = np.array(global_scene[comp].load())
		swath.append(temp[:2030, :1350])

	swath.append(latitude[:2030, :1350])
	swath.append(longitude[:2030, :1350])

	return np.array(swath)

def get_swath_rgb(radiance_filename, composite='true_color_uncorrected'):
	"""
	:param radiance_filename: MOD02 filename
	:return visible RGB channels: numpy.ndarray of size (3, 2030, 1350) 
	Uses the satpy Scene reader with the modis-l1b files. Issues reading files might be due to pyhdf not being
	installed - otherwise try pip install satpy[modis_0l1b]
	Creates a scene with the MOD02 and MOD03 files, and extracts the RGB channels from the two visible MODIS bands.
	"""

	# find a corresponding geolocational (MOD03) file for the provided radiance (MOD02) file
    geoloc_filename = find_matching_geoloc_file(radiance_filepath)

	global_scene = Scene(reader='modis_l1b', filenames=[radiance_filename, geoloc_filename])

	# load it in, make sure resolution is 1000 to match our other datasets
	global_scene.load([composite], resolution=1000)

	rgb = np.array(global_scene[composite])[:,:2030,:1350]

	return rgb