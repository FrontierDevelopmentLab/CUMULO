import numpy as np
import sys

import netCDF4 as nc4

def copy_dataset(original_filename, copy_filename, status="daylight", deep=True):
    
    with nc4.Dataset(original_filename, 'r') as original:

        with nc4.Dataset(copy_filename,'w', format='NETCDF4') as copy:

            block_list = [(copy, original)]
            
            # create groups if deep
            if deep:
                for name, group in original.groups.items():
                    new_group = copy.createGroup(name)
                    block_list.append([new_group, group])

            # copy global attributes
            copy.setncatts({a : original.getncattr(a) for a in original.ncattrs()})
            copy.status_flag = status

            for new_block, block in block_list:

                # copy dimensions
                for name, dim in block.dimensions.items():
                    new_block.createDimension(name, len(dim) if not dim.isunlimited() else None)

                # Copy variables
                for name, var in block.variables.items():
                    new_var = new_block.createVariable(name, var.datatype, var.dimensions)
                    
                    # Copy variable attributes
                    new_var.setncatts({a : var.getncattr(a) for a in var.ncattrs()})

if __name__ == "__main__":

    # swath_path = sys.argv[1]

    swath_shape = (1354, 2030)

    # create a copy of dataset
    copy_dataset = copy_dataset("cumulo.nc", "copy.nc")

        
    
