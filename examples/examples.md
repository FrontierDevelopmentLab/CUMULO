# Usage Examples

### Reading NETCDF files

1. To list the structure (variables, dimensions and descriptions) of the netcdf file ‘A2008DDD.HHMM.nc’, run:

```bash
ncdump -h <AYYYYDDD.HHMM.nc>
```

2. To visualize the content of the netcdf file ‘A2008DDD.HHMM.nc’, we suggest using [**Ncview**](http://meteora.ucsd.edu/~pierce/ncview_home_page.html) or [**Panoply**](https://www.giss.nasa.gov/tools/panoply/download/).

3. To load a netcdf file and get a variable as a masked numpy.ndarray in python, run:

```python

import netCDF4 as nc4

file = nc4.Dataset(‘A2008DDD.HHMM.nc’, 'r', format='NETCDF4')
variable_content = file.variables['variable_name'][:]

```

### Loading CUMULO

```python

```