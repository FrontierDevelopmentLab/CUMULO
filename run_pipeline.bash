#!/usr/bin/env bash
find ../DATA/aqua-data/level_1/2008/01 -type f | grep "MYD021KM" | xargs --max-procs=64 -n 1 python -W ignore pipeline.py
