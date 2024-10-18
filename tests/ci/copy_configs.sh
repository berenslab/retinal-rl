#!/bin/bash
#===============================================================================
# Description: Copies the config files from the resources dir to the config dir.
#              Thus creates the basic structure needed to setup or run
#              experiments.
#
# Usage:
#   tests/ci/copy_configs.sh
#   (run from top level directory!)
#===============================================================================

cp -r resources/config_templates/* config/