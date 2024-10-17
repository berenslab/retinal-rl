#!/bin/bash
# first argument denotes container

for file in config/user/experiment/*.yaml; do
    experiment=$(basename "$file" .yaml)
    apptainer exec "$1" \
    python main.py +experiment="$experiment" command=scan system.device=cpu
done