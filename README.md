# CUMULO

Code sourse of CUMULO, a benchmark dataset for training and evaluating global cloud classification models. 
It merges two satellite products from the [A-train constellation](https://atrain.nasa.gov/): 
the [Moderate Resolution Imaging Spectroradiometer (MODIS) from Aqua satellite](https://modis.gsfc.nasa.gov/about/) and the [2B-CLDCLASS-LIDAR product](http://www.cloudsat.cira.colostate.edu/data-products/level-2b/2b-cldclass-lidar) derived from the combination of CloudSat Cloud Profiling Radar (CPR) and CALIPSO Cloud‐Aerosol Lidar with Orthogonal Polarization (CALIOP).


### Installation

```bash
pip install gcsfs
conda install -c conda-forge pyhdf  #The pip install's wheels are broken at time of writing
pip install satpy
pip install satpy[modis_l1b]
pip install -r requirements.txt
```

### File Content

To get an overview of CUMULO's variables and their descriptions, run

```bash
ncdump -h netcdf/cumulo.nc
```

### Acknowledgments

This work is the result of the 2019 ESA [Frontier Development Lab](https://fdleurope.org/) Atmospheric Phenomena andClimate Variability challenge.  We are grateful to all organisers, mentors and sponsors for providing us this opportunity. We thank Google Cloud for providing computing and storage resources to complete this work.

### Cite

[Original paper](https://arxiv.org/abs/1911.04227)