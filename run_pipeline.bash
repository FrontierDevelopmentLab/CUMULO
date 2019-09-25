#!/usr/bin/env bash
for month in 01 02 03 04 05 06
do
	find /mnt/disks/disk1/aqua-data/level_1/2008/$month/ -type f | grep "MYD021KM" | xargs --max-procs=65 -n 1 python -W ignore pipeline.py /mnt/disks/disk1/2008/$month/
done
