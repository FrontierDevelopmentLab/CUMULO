import numpy as np
import os
import sys

import netCDF4 as nc4

from src.track_alignment import map_and_reduce
from src.utils import get_file_time_info, minutes_since    

swath_channels = ['EV_250_Aggr1km_RefSB_1', 'EV_250_Aggr1km_RefSB_2', 'EV_1KM_Emissive_29', 'EV_1KM_Emissive_33', 'EV_1KM_Emissive_34', 'EV_1KM_Emissive_35', 'EV_1KM_Emissive_36', 'EV_1KM_RefSB_26', 'EV_1KM_Emissive_27', 'EV_1KM_Emissive_20', 'EV_1KM_Emissive_21', 'EV_1KM_Emissive_22', 'EV_1KM_Emissive_23', 'Latitude', 'Longitude', 'Cloud_Water_Path', 'Cloud_Optical_Thickness', 'Cloud_Effective_Radius', 'Cloud_Phase_Optical_Properties', 'cloud_top_pressure_1km', 'cloud_top_height_1km', 'cloud_top_temperature_1km', 'cloud_emissivity_1km', 'surface_temperature_1km', 'Cloud_Mask']

layer_info_channels = {
    "type-layer" : 'Cloud_Layer_Type', 
    "base-layer" : 'Cloud_Layer_Base',
    "top-layer" : 'Cloud_Layer_Top', 
    "type-quality" : 'Cloud_Type_Quality',
    "precip-flag" : 'Precipitation_Flag',
}


channel_params = { # offset, scale_factor, min_value, max_value
    'Cloud_Water_Path' : [0., 1., 0, 10000], 
    'Cloud_Optical_Thickness' : [0., 0.00999999977648258, 0, 10000], 
    'Cloud_Effective_Radius' : [0., 0.00999999977648258, 0, 10000], 
    'Cloud_Phase_Optical_Properties' : [0., 1., 0, 4], 
    'cloud_top_pressure_1km' : [0., 0.100000001490116, 10, 11000], 
    'cloud_top_height_1km' : [0., 1., 0, 18000], 
    'cloud_top_temperature_1km' : [-15000, 0.00999999977648258, 0, 20000], 
    'cloud_emissivity_1km' : [0., 0.00999999977648258, 0, 100], 
    'surface_temperature_1km' : [-15000, 0.00999999977648258, 0, 20000], 
}

def copy_dataset(original_filename, copy_filename, deep=True):
    
    with nc4.Dataset(original_filename, 'r') as original:

        copy = nc4.Dataset(copy_filename, 'w', format='NETCDF4')

        block_list = [(copy, original)]
        variables = {}

        # create groups if deep
        if deep:
            for name, group in original.groups.items():
                new_group = copy.createGroup(name)
                block_list.append([new_group, group])

        # copy global attributes
        copy.setncatts({a : original.getncattr(a) for a in original.ncattrs()})

        for new_block, block in block_list:

            # copy dimensions
            for name, dim in block.dimensions.items():
                new_block.createDimension(name, len(dim) if not dim.isunlimited() else None)

            # Copy variables
            for name, var in block.variables.items():

                new_var = new_block.createVariable(name, var.datatype, var.dimensions)
                
                # Copy variable attributes
                new_var.setncatts({a : var.getncattr(a) for a in var.ncattrs()})

                variables[name] = new_var

    return copy, variables

def fill_dataset(dataset, variables, swath, layer_info, minutes, status="daylight", deep=True):

    shape = swath[0].shape

    for i, channel in enumerate(swath_channels):

        try:

            invalid_mask = (swath[i] < channel_params[channel][2]) | (swath[i] > channel_params[channel][3])

            # scale values
            swath[i] = (swath[i] + channel_params[channel][0]) * channel_params[channel][1]

            # fill values outside valid range with fillvalue            
            swath[i][invalid_mask] = variables[channel]._FillValue

        except KeyError:
            pass

        variables[channel][0] = swath[i].T

    for info_name, channel in layer_info_channels.items():

        # map data to swath format
        if info_name == "precip-flag":
            # precipitation flag is not available per layer
            info = np.full(shape, variables[channel]._FillValue)
            map_and_reduce(layer_info["mapping"], layer_info[info_name], info, layer_info["width-range"])
            info = info.T

        else:

            info = np.full((*shape, 10), variables[channel]._FillValue)
            map_and_reduce(layer_info["mapping"], layer_info[info_name], info, layer_info["width-range"])
            info = info.transpose(1, 0, 2)

            # correct values, from [1, 8] to [0, 7]
            if info_name == "type-layer":
                info[info > -1] -= 1
                info[info < 0] = variables[channel]._FillValue

        variables[channel][0] = info

    # set global variables and attributes
    dataset.status = status
    dataset["Time"][0] = minutes

def load_npys(swath_path, layer_info_dir="layer-info"):

    dirname, filename = os.path.split(swath_path)

    swath = np.load(swath_path)
    layer_info_dict = np.load(os.path.join(dirname, layer_info_dir, filename)).item()

    return swath, layer_info_dict


if __name__ == "__main__":

    swath_path = sys.argv[1]
    save_dir = sys.argv[2]

    swath, layer_info = load_npys(swath_path)

    # get time info
    year, abs_day, hour, minute = get_file_time_info(swath_path)

    # create a copy of reference dataset
    copy_name = "A{}.{}.{}{}.nc".format(year, abs_day, hour, minute)
    copy, variables = copy_dataset("netcdf/cumulo.nc", copy_name)

    # determine swath status from directory hierarchy
    status = "corrupt"
    if "daylight" in swath_path:
        status = "daylight"
    elif "night" in swath_path:
        status = "night"

    # convert npy to nc
    minutes_since_2008 = minutes_since(int(year), int(abs_day), int(hour), int(minute))
    fill_dataset(copy, variables, swath, layer_info, minutes_since_2008, status)

    copy.close()
