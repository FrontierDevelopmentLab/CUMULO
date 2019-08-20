#!/usr/bin/env bash
find /mnt/disks/disk10/aqua-data/level_1/2008/08 -type f | grep "MYD021KM" | xargs --max-procs=64 -n 1 python -W ignore pipeline.py
