#!/bin/bash
# $1 choses the sif container, $2 defines whether pylint is run on all or only the changed files

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