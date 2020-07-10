# Usage Examples

## Reading NETCDF files

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

## CUMULO for Machine Learning

Check out [loader.py](src/loader.py) for loading utils.
CUMULO's variables are categorized into:

1. geographic coordinates

```python
coordinates = ['latitude', 'longitude']
```

2. calibrated radiances (training features)

```python
radiances = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33', 'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26', 'ev_1km_emissive_27', 'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
```

3. computed cloud properties (derived from radiances)

```python
properties = ['cloud_water_path', 'cloud_optical_thickness', 'cloud_effective_radius', 'cloud_phase_optical_properties', 'cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature', 'cloud_emissivity', 'surface_temperature']
```

4. cloud binary mask (telling whether a pixel is certainly cloudy or not)

```python
rois = 'cloud_mask'
```

5. annotations and cloud information (from CloudSat, available only along the track of the satellite) 

```python
labels = 'cloud_layer_type'
additional_information = ['cloud_layer_base', 'cloud_layer_top', 'cloud_type_quality', 'precipitation_flag']
```

IMPORTANT: 

> All variables containing _layer_ in their name have an additional vertical dimension (latitude - longitude - cloud layer). Therefore, each 2D-pixel can take multiple values.
> These variables are defined on up to 10 different vertical layers of clouds.
> Distinct cloud vertical layers are identified by splitting cloud clusters with hydrometeor-free separation of at least 480 m. Because spotted clouds obviously vary over space and time both in type and quantity, layers are not predefined intervals of fixed size over the height, but their number and thickness vary over the pixels. 


In [our work](https://arxiv.org/abs/1911.04227), we classified clouds by retaining for each pixel the most frequent label from *cloud_layer_type* but there could be better choices (e.g., using the distribution of labels for each pixel, or weighting labels by layer thickness).

### Running Baselines

#### Tile extraction
The provided methods (iResNet and LightGBM) are applied on 3x3 tiles extracted from the whole images using the following script.

``` bash
python netcdf/nc_tile_extractor.py
```

Labeled tiles are sampled around each labeled pixel of an image and an equal amount of unlabeled tiles is sampled uniformly on the remaining cloudy portions of the image.

#### ML Baselines

##### LightGBM

1. The jupyter notebook [training](lgbm.ipynb) provides the code for training a LightGBM model. See [doc](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) for installation.

2. The script [predicting](lightgbm_predict.py) provides the code for predicting over the whole swath using the trained model. 
As the model takes as input 3x3 tiles, it is applied on the 2030x1354 swath sequentially and without overlappings.


##### iResNet

The provided code is an adaptation of [Invertible Residual Networks, ICML 2019](https://github.com/jhjacobsen/invertible-resnet). 

1. The script [training](iresnet_training.py) provides the code for training a hybrid iResNet on CUMULO.

1. The script [predicting](iresnet_predict.py) provides the code for predicting over the whole swath using the trained model. 
As the model takes as input 3x3 tiles, it is applied on the 2030x1354 swath sequentially and without overlappings.
