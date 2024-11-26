#!/bin/bash
#===============================================================================
# Description: "Scans" all experiments, that means tries to read the config and
#              build the model, than prints the summary.
#
# Arguments:
#   $1 - Path to Singularity (.sif) container
#
# Usage:
#   tests/ci/run_experiments.sh container.sif
#   (run from top level directory!)
#
# Dependencies:
#   - Singularity/Apptainer + container
#   - YAML configuration files in correct directory
#===============================================================================

for file in config/user/experiment/*.yaml; do
    experiment=$(basename "$file" .yaml)
    singularity exec "$1" \
    python main.py +experiment="$experiment" command=scan system.device=cpu
done