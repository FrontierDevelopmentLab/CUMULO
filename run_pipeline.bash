#!/usr/bin/env bash
find ./fdl-modis-l1/ | grep MOD021KM | xargs --max-procs=60 -n 1 python pipeline.py
