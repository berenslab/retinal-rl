#!/bin/bash
#===============================================================================
# Description: Runs pylint either on all Python files or only on changed files
#             compared to master branch using a specified Singularity container
#
# Arguments:
#   $1 - Path to Singularity (.sif) container
#   $2 - Optional: "--all" to run on all files, otherwise runs only on changed files
#
# Usage:
#   tests/ci/run_pylint.sh container.sif         # Lint only changed Python files
#   tests/ci/run_pylint.sh container.sif --all   # Lint all Python files
#   (run from top level directory!)
#
# Dependencies:
#   - Singularity/Apptainer
#   - Container must have pylint installed
#===============================================================================

if [ "$2" = "--all" ]; then
    apptainer exec "$1" pylint .
else
    changed_files=$(git diff --name-only origin/master...HEAD -- '*.py')
    if [ -n "$changed_files" ]; then
        apptainer exec "$1" pylint $(git diff --name-only origin/master...HEAD -- '*.py')
    else
        echo "No .py files changed"
    fi
fi