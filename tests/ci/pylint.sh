#!/bin/bash
# $1 choses the sif container, $2 defines whether pylint is run on all or only the changed files

if [ "$2" = "--all" ]; then
    apptainer exec "$1" pylint .
else
    apptainer exec "$1" pylint $(git diff --name-only origin/master...HEAD -- '*.py')
fi