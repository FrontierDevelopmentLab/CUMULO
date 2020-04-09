#!/usr/bin/env bash
month="05";
year="2009";
find "/mnt/modisaqua/$year/MODIS/data/MYD021KM/collection61/$year/$month/" -type f | grep "MYD021KM" | xargs --max-procs=10 -n 1 python -W ignore pipeline.py "../disks/disk1/$year/$month/"
# cat "missing$month.txt" | xargs --max-procs=10 -n 1 python -W ignore pipeline.py "../disks/disk2/$year/$month/"
