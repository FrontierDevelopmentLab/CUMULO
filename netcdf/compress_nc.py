import netCDF4 as nc4
import os
import sys

def compress_dataset(original_filename, copy_filename):
    
    with nc4.Dataset(original_filename, 'r') as original:

        with nc4.Dataset(copy_filename, 'w', format='NETCDF4') as copy:

            block_list = [(copy, original)]

            for name, group in original.groups.items():
                new_group = copy.createGroup(name)
                block_list.append((new_group, group))

            # copy global attributes
            copy.setncatts({a : original.getncattr(a) for a in original.ncattrs()})

            for new_block, block in block_list:

                # copy dimensions
                for name, dim in block.dimensions.items():
                    new_block.createDimension(name, len(dim) if not dim.isunlimited() else None)

                # Copy variables
                for name, var in block.variables.items():

                    new_var = new_block.createVariable(name, var.datatype, var.dimensions, zlib=True)
                    
                    # Copy variable attributes
                    new_var.setncatts({a : var.getncattr(a) for a in var.ncattrs()})

                    # Copy variable content
                    new_var[:] = var[:]

if __name__ == "__main__":

    swath_path = sys.argv[1]

    copy_name = swath_path.replace(".nc", "-compressed.nc")
    compress_dataset(swath_path, copy_name)

    # remove original file
    os.remove(swath_path)
    os.rename(copy_name, swath_path)
