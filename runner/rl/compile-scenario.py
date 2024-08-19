import sys
import argparse
import os

from retinal_rl.rl.scenarios.preload import (
    preload_apples,
    preload_cifar10,
    preload_gabors,
    preload_mnist,
    preload_obstacles,
    cache_dir,
)
from retinal_rl.rl.scenarios.make import make_scenario, SCENARIO_YAML_DIR


def make_parser():
    # Initialize parser
    parser = argparse.ArgumentParser(
        description=f"""Utility to construct scenarios for retinal-rl, by
        merging YAML files from the '{SCENARIO_YAML_DIR}' directory into a
        specification for a scenario. By default the scenario name is the
        concatenation of the names of the yaml files, but can be set with the
        --name flag. The results are saved in the 'scenarios' directory. Before
        running the first time one should use the --preload flag to download the
        necessary resources into the '{cache_dir}' directory.
        """,
        epilog="Example: python -m exec.compile-scenario gathering apples",
    )
    # Positional argument for scenario yaml files (required, can be multiple)
    parser.add_argument(
        "yamls",
        nargs="*",
        help="""Names of the component yaml files (without extension) for the
        desired scenario. For conflicting fields, the last file will take precedence.""",
    )
    # Argument for optional scneario name
    parser.add_argument(
        "--name",
        help="Desired name of scenario",
    )
    # Add option to run preload
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload resources",
    )
    # List the contents of the scenario yaml directory
    parser.add_argument(
        "--list_yamls",
        action="store_true",
        help="List available scenario yamls",
    )
    return parser


def main():

    # Parse args
    argv = sys.argv[1:]
    parser = make_parser()
    args = parser.parse_args(argv)

    # Check preload flag
    if args.preload:
        preload_apples()
        preload_obstacles()
        preload_gabors()
        preload_mnist()
        preload_cifar10()
        # exit after preloading
        return 0

    if args.list_yamls:
        print(f"Listing contents of {SCENARIO_YAML_DIR}:")
        for flnm in os.listdir(SCENARIO_YAML_DIR):
            print(flnm)
        return 0

    # positional arguments
    if len(args.yamls) > 0:
        make_scenario(args.yamls, args.name)
    else:
        # no positional, warn and exit
        print("No yaml files provided. Nothing to do.")


if __name__ == "__main__":
    sys.exit(main())
