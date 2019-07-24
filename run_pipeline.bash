#!/usr/bin/env bash
find ../DATA/raw/190723_unsup_pipeline/ -type f | grep "MOD021KM" | xargs --max-procs=64 -n 1 python unsupervised_pipeline_sequential.py