#!/usr/bin/env bash
month="04";
#find "/mnt/modisaqua/2016/MODIS/data/MYD021KM/collection61/2016/$month/" -type f | grep "MYD021KM" | xargs --max-procs=10 -n 1 python -W ignore pipeline.py "../disks/disk2/2016/$month/"
cat "missing$month.txt" | xargs --max-procs=10 -n 1 python -W ignore pipeline.py "../disks/disk2/2016/$month/"
