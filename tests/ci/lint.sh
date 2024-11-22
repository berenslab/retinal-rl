#===============================================================================
# Description: Runs ruff either on all Python files or only on changed files
# compared to master branch using a specified Singularity container
# Runs both code formatting and linter, but additional arguments (other than 
# --all and --fix) will only apply to linter.
#
# Arguments:
# $1 - Path to Singularity (.sif) container
# $2 - Optional: "--all" to run on all files, otherwise runs only on changed files
# $3+ - Optional: Additional arguments passed directly to ruff
#
# Usage:
# tests/ci/lint.sh container.sif # Lint only changed Python files
# tests/ci/lint.sh container.sif --all # Lint all Python files
# tests/ci/lint.sh container.sif --all --select E501,F401 # Lint all files with specific rules
# tests/ci/lint.sh container.sif "" --fix # Fix changed files
# (run from top level directory!)
#
# Dependencies:
# - Singularity/Apptainer + container
# - Container must have ruff installed
#===============================================================================

# Store the container path
CONTAINER="$1"
shift

# Check or fix
check="--check"
if [[ "$@" == *"--fix"* ]]; then
    check=""
fi

# Check if --all flag is present
if [ "$1" = "--all" ]; then
    changed_files="."
    # Remove --all from arguments
    shift
else
    # Get changed Python files
    changed_files=$(tests/ci/changed_py_files.sh)
fi

if [ -n "$changed_files" ]; then
    # Format
    singularity exec "$CONTAINER" ruff format $changed_files $check
    # Run ruff on changed files with any remaining arguments
    singularity exec "$CONTAINER" ruff check $changed_files "$@"
else
    echo "No .py files changed"
fi
