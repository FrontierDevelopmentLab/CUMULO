# Data Pipeline

## Introduction

To feed the AI and causal inference work downstream, we need to create a series of inter-linked scripts that take raw data in-bucket, and produce arrays and outputs ready for model ingestion.

The major stages are:

1. Pulling the data from the bucket to a pipeline instance.

2. Extracting information from the MODIS `.hdf` files, using `satpy`, to produce usable arrays.
3. Correcting for pixel-scale `NaN` artefacts inherited from the raw data.
4. Extracting tiles from the processed arrays, ready for model ingestion.
5. Pushing the arrays up to bucket, for project-wide consumption.

### Requirements

As this repository involves several different steps, the list of requirements is more extensive. In addition, the `satpy` installation can be very temperamental, so using environments is highly recommended.

```bash
#To create the new env

conda create -n pipeline python=3.7
conda activate pipeline

#Installation
pip install gcsfs
pip install satpy
pip install satpy[modis_l1b]
conda install -c conda-forge pyhdf  #The pip install's wheels are broken at time of writing
```

There are also standard data science libraries used (`Numpy`, `Pandas` etc...). These are not fully described for brevity.

### Gotchas

#### `pyhdf` Permissions

There is a known Gotcha with file permissions for installing `pyhdf`. If you end up with errors like:

```sh
[Errno 13] Permission denied: '/home/jupyter/.cph_tmpza9rngq5'
[Errno 13] Permission denied: '/home/jupyter/.cph_tmpbt192k6i'
[Errno 13] Permission denied: '/home/jupyter/.cph_tmpvzn6qtgo'
[Errno 13] Permission denied: '/home/jupyter/.cph_tmpdrrqg__5'
```

run `sudo chown -R USERNAME:USERNAME /home/USERNAME/` replacing USERNAME with your user. Run the install again -  you should have a clean run.


