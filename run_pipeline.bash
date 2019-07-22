#!/usr/bin/env bash
find ../DATA/modis-l1/ -type f | grep "MOD021KM | xargs --max-procs=24 -n 1 python unsupervised_pipeline.py