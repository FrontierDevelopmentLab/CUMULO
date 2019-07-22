from satpy import Scene
import glob
import numpy as np
import pickle
import pdb


def run(path, save_dir='./test_dump/'):
	"""
	:param path: str, path to hdf directory
	:param save_dir: optional - path to target file destination
	:return:
	Given a path in a string format and a save dir (optional otherwise dumps to a test_dump dir) outputs a pickled
	array saved with the identifying string and date.
	"""
	filenames = get_modis_filenames(path)
	for match in filenames:
		parts = match[0].split('.')
		dump_array = get_arrays(match)
		dump_filename = parts[2].replace('A', '') + '_' + parts[3] + '.pkl'
		with open(save_dir + dump_filename, 'wb') as f:
			pickle.dump(dump_array, f)


def get_modis_filenames(path):
	"""
	:param path: Path to the directory of MOD02 files
	:return: The filenames for the matched counterpart files
	get the filenames for the radiances and the geolocation information
	"""
	radiance_filenames = glob.glob(path + '*D021KM*.hdf')
	matches = []
	for radiance_filename in radiance_filenames:
		identifier = radiance_filename.split("/")[-1].split('A')[1].split('.')[1]
		geoloc_file = glob.glob(path + '*D03*.{}.*.hdf'.format(identifier))[0]
		matches.append([radiance_filename, geoloc_file])
	# returns in the expected format, geoloc then radiance in a list for all MODIS files within the path
	return matches


def get_arrays(files):
	"""
	:param files: list of nested paired MOD02 and MOD03 files
	:return: numpy array of nested bands per MODIS02 file
	Uses the satpy Scene reader with the modis-l1b files. Issues reading files might be due to pyhdf not being
	installed - otherwise try pip install satpy[modis0l1b]
	"""
	composite = ['1', '2', '29', '33', '34', '35', '36', '26', '27', '20', '21', '22', '23']
	# load the global scene using satpy
	# expects filenames to be radiance then geoloc
	# filenames must be in specific format, chop off beginning to work
	global_scene = Scene(reader='modis_l1b', filenames=files)

	# print available_composites
	# expects the composite to be in a list
	global_scene.load(composite, resolution=1000)
	
	# for database structuring
	global_scene.load(['latitude', 'longitude'], resolution=1000)
	latitude = np.array(global_scene['latitude'].load())
	longitude = np.array(global_scene['longitude'].load())
	dump_array = []
	# note that we only take til 1350 to avoid any bowtie effects
	for comp in composite:
		temp = np.array(global_scene[comp].load())
		dump_array.append(temp[:, :1350])
	dump_array.append(latitude[:, :1350])
	dump_array.append(longitude[:, :1350])

	return np.array(dump_array)
