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

if [ "$1" == "" ]; then
    SINGULARITY_PREFIX=""
else
    SINGULARITY_PREFIX="singularity exec $1"
fi

for file in config/user/experiment/*.yaml; do
    experiment=$(basename "$file" .yaml)
    $SINGULARITY_PREFIX python main.py +experiment="$experiment" command=scan system.device=cpu
done