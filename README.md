<img src="https://github.com/FrontierDevelopmentLab/CUMULO/blob/master/docs/images/cumulo.png" width="300">

a benchmark dataset for training and evaluating global cloud classification models. 
It merges two satellite products from the [A-train constellation](https://atrain.nasa.gov/): 
the [Moderate Resolution Imaging Spectroradiometer (MODIS) from Aqua satellite](https://modis.gsfc.nasa.gov/about/) and the [2B-CLDCLASS-LIDAR product](http://www.cloudsat.cira.colostate.edu/data-products/level-2b/2b-cldclass-lidar) derived from the combination of CloudSat Cloud Profiling Radar (CPR) and CALIPSO Cloud‐Aerosol Lidar with Orthogonal Polarization (CALIOP).

[FULL README](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AAAeTWF21WGZ7-y9MpSiL9P3a/CUMULO?dl=0&preview=README.pdf&subfolder_nav_tracking=1)

# Dataset

The dataset is hosted [here](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AADq2lk4u7k961Qa31FwIDEpa?dl=0).
It contains over 300k annotated multispectral images at 1km x 1km resolution, providing daily full coverage of the Earth for 2008, 2009 and 2016.

## Download

#### Option 1: syncing with your DropBox Account
1. add [CUMULO](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AADq2lk4u7k961Qa31FwIDEpa?dl=0) to your DropBox account
2. use [rclone](https://rclone.org/dropbox/) for syncing it on your machine

#### Option 2: direct download
1. use one of these download [scripts](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AACRmgYpsWtw6qEa_JKD9Hp_a/CUMULO/download-scripts_2008?dl=0&subfolder_nav_tracking=1)

### File Format

Data is stored in **Network Common Data Form (NetCDF)** following this [convention](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html).

There is 1 NetCDF file per swath of 1354x2030 pixels, 1 every 5 minutes, named:

```
filename = AYYYYDDD.HHMM.nc

YYYY => year
DDD => absolute day since 01.01.YYYY 
HH => hour of day
MM => minutes    
```

### File Content

To see the variables available for a netcdf file and their description, run: 

```bash
ncdump -h netcdf/cumulo.nc
```

## Code Source

1. The script [pipeline.py](pipeline.py) extracts one CUMULO's swath (as a netcdf file) from the corresponding MODIS' MYD02, MYD03, MYD06 and MYD35 files, and CloudSat's CS_2B-CLDCLASS and/or CS_2B-CLDCLASS-LIDAR files.

```python
python3 pipeline <save-dir> <myd02-filename>
```

2. [src/](src/) contains the code source for extracting the different CUMULO's features, for alignment them and for completing the missing values when possible.

### Dependencies

```bash
pip install gcsfs
conda install -c conda-forge pyhdf  #The pip install's wheels are broken at time of writing
pip install satpy
pip install satpy[modis_l1b]
pip install -r requirements.txt
```

## Machine Learning Baselines
Examples for training models on CUMULO are provided [here](ml-examples/).

## Cite
If you find this work useful, please cite the [original paper](https://arxiv.org/abs/1911.04227):

```
@article{zantedeschi2019cumulo,
        title={Cumulo: A Dataset for Learning Cloud Classes},
        author={Zantedeschi, Valentina and Falasca, Fabrizio and Douglas, Alyson and Strange, Richard and Kusner, Matt J and Watson-Parris, Duncan},
        journal={arXiv preprint arXiv:1911.04227},
        year={2019}}
```

## Acknowledgments

This work is the result of the 2019 ESA [Frontier Development Lab](https://fdleurope.org/) Atmospheric Phenomena and Climate Variability challenge. 
We are grateful to all organisers, mentors and sponsors for providing us this opportunity. We thank Google Cloud for providing computing and storage resources to complete this work.
