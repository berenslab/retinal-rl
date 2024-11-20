#!/bin/bash
#===============================================================================
# Description: Builds the gathering apples scenario used for tests
#
# Arguments:
#   $1 - Path to Singularity (.sif) container
#
# Usage:
#   tests/ci/build_scenario.sh container.sif
#   (run from top level directory!)
#===============================================================================

apptainer exec "$1" python -m doom_creator.compile_scenario gathering apples