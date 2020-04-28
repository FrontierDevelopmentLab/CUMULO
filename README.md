<img src="https://github.com/FrontierDevelopmentLab/CUMULO/blob/master/docs/images/cumulo.png" width="300">

a benchmark dataset for training and evaluating global cloud classification models. 
It merges two satellite products from the [A-train constellation](https://atrain.nasa.gov/): 
the [Moderate Resolution Imaging Spectroradiometer (MODIS) from Aqua satellite](https://modis.gsfc.nasa.gov/about/) and the [2B-CLDCLASS-LIDAR product](http://www.cloudsat.cira.colostate.edu/data-products/level-2b/2b-cldclass-lidar) derived from the combination of CloudSat Cloud Profiling Radar (CPR) and CALIPSO Cloud‐Aerosol Lidar with Orthogonal Polarization (CALIOP).

[FULL README](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AAAeTWF21WGZ7-y9MpSiL9P3a/CUMULO?dl=0&preview=README.pdf&subfolder_nav_tracking=1)

## Dataset

The dataset is hosted [here](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AADq2lk4u7k961Qa31FwIDEpa?dl=0).
It contains over 100k annotated multispectral images at 1km x 1km resolution, providing daily coverage of the Earth for 2008. Years 2009 and 2016 are coming soon.

### Download

#### Option 1: syncing with your DropBox Account
1. add [CUMULO](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AADq2lk4u7k961Qa31FwIDEpa?dl=0) to your DropBox account
2. use [rclone](https://rclone.org/dropbox/) for syncing it on your machine

#### Option 2: direct download
1. use one these download [scripts](https://www.dropbox.com/sh/6gca7f0mb3b0ikz/AACJu8tYZpREqL704LZ2XVQ9a/CUMULO/download-scripts?dl=0&subfolder_nav_tracking=1)


### File Content

To get an overview of CUMULO's variables and their descriptions, run

```bash
ncdump -h netcdf/cumulo.nc
```

## Code Source

### Dependencies

```bash
pip install gcsfs
conda install -c conda-forge pyhdf  #The pip install's wheels are broken at time of writing
pip install satpy
pip install satpy[modis_l1b]
pip install -r requirements.txt
```

## Acknowledgments

This work is the result of the 2019 ESA [Frontier Development Lab](https://fdleurope.org/) Atmospheric Phenomena and Climate Variability challenge. 
We are grateful to all organisers, mentors and sponsors for providing us this opportunity. We thank Google Cloud for providing computing and storage resources to complete this work.

## Cite
If you find this work useful, please cite the [original paper](https://arxiv.org/abs/1911.04227):

```
@article{zantedeschi2019cumulo,
        title={Cumulo: A Dataset for Learning Cloud Classes},
        author={Zantedeschi, Valentina and Falasca, Fabrizio and Douglas, Alyson and Strange, Richard and Kusner, Matt J and Watson-Parris, Duncan},
        journal={arXiv preprint arXiv:1911.04227},
        year={2019}}
```
