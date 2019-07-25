#!/usr/bin/env bash
find /mnt/disks/disk4/l1_aqua/ -type f | grep "MYD021KM" | xargs --max-procs=96 -n 1 python semisupervised_pipeline_random.py
