find ../DATA/raw/190723_unsup_pipeline/ -type f | grep "MOD021KM" | xargs --max-procs=24 -n 1 python unsupervised_pipeline.py
