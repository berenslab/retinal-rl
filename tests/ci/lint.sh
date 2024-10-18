#!/bin/bash
#===============================================================================
# Description: Runs ruff either on all Python files or only on changed files
#             compared to master branch using a specified Singularity container
#
# Arguments:
#   $1 - Path to Singularity (.sif) container
#   $2 - Optional: "--all" to run on all files, otherwise runs only on changed files
#
# Usage:
#   tests/ci/lint.sh container.sif         # Lint only changed Python files
#   tests/ci/lint.sh container.sif --all   # Lint all Python files
#   (run from top level directory!)
#
# Dependencies:
#   - Singularity/Apptainer + container
#   - Container must have ruff installed
#===============================================================================

if [ "$2" = "--all" ]; then
    apptainer exec "$1" ruff check .
else
    changed_files=$(git diff --name-only origin/master...HEAD -- '*.py')
    if [ -n "$changed_files" ]; then
        apptainer exec "$1" ruff check $(git diff --name-only origin/master...HEAD -- '*.py')
    else
        echo "No .py files changed"
    fi
fi