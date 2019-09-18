#!/usr/bin/env bash
for month in 01 02 03 04 05 06 07 08 09 10 11 12
do
	find /mnt/disks/disk1/aqua-data/level_1/2008/$month/ -type f | grep "MYD021KM" | xargs --max-procs=80 -n 1 python -W ignore pipeline.py /mnt/disks/disk1/2008/$month/
done
