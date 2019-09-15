#!/usr/bin/env bash
find ../DATA/aqua-data/level_1/2008/01/01/ -type f | grep "MYD021KM" | xargs --max-procs=40 -n 1 python -W ignore pipeline.py ../DATA/data-processed/2008/01/01/
