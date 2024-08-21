import sys
import argparse
import os

from doom_creator.util.preload import preload
from doom_creator.util.preload import ImageDataType

from doom_creator.util.make import make_scenario
from doom_creator.util.directories import Directories


def make_parser():
    # Initialize parser
    Directories()
    parser = argparse.ArgumentParser(
        description=f"""Utility to construct scenarios for retinal-rl, by
        merging YAML files from the '{Directories().SCENARIO_YAML_DIR}' directory into a
        specification for a scenario. By default the scenario name is the
        concatenation of the names of the yaml files, but can be set with the
        --name flag. The results are saved in the 'scenarios' directory. Before
        running the first time one should use the --preload flag to download the
        necessary resources into the '{Directories().CACHE_DIR}' directory.
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

    directories = Directories()
    # Check preload flag
    if args.preload:
        preload(ImageDataType.APPLES, directories.TEXTURES_DIR, directories.ASSETS_DIR)
        preload(ImageDataType.OBSTACLES, directories.TEXTURES_DIR, directories.ASSETS_DIR)
        preload(ImageDataType.GABORS, directories.TEXTURES_DIR, directories.ASSETS_DIR)
        preload(ImageDataType.MNIST, directories.TEXTURES_DIR)
        preload(ImageDataType.CIFAR10, directories.TEXTURES_DIR)
        # exit after preloading
        return 0

    if args.list_yamls:
        print(f"Listing contents of {directories.SCENARIO_YAML_DIR}:")
        for flnm in os.listdir(directories.SCENARIO_YAML_DIR):
            print(flnm)
        return 0

    # positional arguments
    if len(args.yamls) > 0:
        make_scenario(args.yamls, directories, args.name)
    else:
        # no positional, warn and exit
        print("No yaml files provided. Nothing to do.")


if __name__ == "__main__":
    sys.exit(main())
