#!/usr/bin/env bash
find /mnt/disks/disk10/aqua-data/level_1/2008/01/ -type f | grep "MYD021KM" | xargs --max-procs=40 -n 1 python -W ignore pipeline.py /mnt/disks/disk10/2008/01/
