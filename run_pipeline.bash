#!/usr/bin/env bash
find dataset-path/ -type f | grep "MYD021KM" | xargs --max-procs=65 -n 1 python -W ignore pipeline.py save-dir
