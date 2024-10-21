#===============================================================================
# Description: Runs ruff either on all Python files or only on changed files
# compared to master branch using a specified Singularity container
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

# Check if --all flag is present
if [ "$1" = "--all" ]; then
    # Remove --all from arguments
    shift
    # Run ruff on all files with any remaining arguments
    apptainer exec "$CONTAINER" ruff check . "$@"
else
    # If first arg isn't --all, put it back in the argument list
    if [ -n "$1" ]; then
        set -- "$1" "$@"
    fi
    
    # Get changed Python files
    changed_files=$(git diff --name-only origin/master...HEAD -- '*.py')
    if [ -n "$changed_files" ]; then
        # Run ruff on changed files with any remaining arguments
        apptainer exec "$CONTAINER" ruff check $(git diff --name-only origin/master...HEAD -- '*.py') "$@"
    else
        echo "No .py files changed"
    fi
fi