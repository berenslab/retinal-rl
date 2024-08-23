import argparse
import os
import sys

from doom_creator.util.directories import Directories
from doom_creator.util.make import make_scenario
from doom_creator.util.preload import preload
from doom_creator.util.texture import TextureType as TType


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
        necessary resources into the --out_dir ('{Directories().CACHE_DIR}').
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
    # Argument for optional scenario name
    parser.add_argument(
        "--name",
        help="Desired name of scenario",
    )
    parser.add_argument(
        "--out_dir",
        default=Directories().CACHE_DIR,
        help="where to store the created scenario",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="source directory of a dataset (for preloading), if you already downloaded it somewhere",
    )
    parser.add_argument(
        "--resource_dir",
        default=Directories().RESOURCE_DIR,
        help="directory where the resources are stored",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Load test split instead of train split",
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

    dirs = Directories(args.out_dir)
    # Check preload flag
    do_load, do_make, do_list = args.preload, len(args.yamls) > 0, args.list_yamls
    if do_load:
        preload(TType.APPLES, dirs.TEXTURES_DIR, dirs.ASSETS_DIR)
        preload(TType.OBSTACLES, dirs.TEXTURES_DIR, dirs.ASSETS_DIR)
        preload(TType.GABORS, dirs.TEXTURES_DIR, dirs.ASSETS_DIR)

        preload(TType.MNIST, dirs.TEXTURES_DIR, args.dataset_dir, train=not args.test)
        preload(TType.CIFAR10, dirs.TEXTURES_DIR, args.dataset_dir, train=not args.test)
    if do_list:
        print(f"Listing contents of {dirs.SCENARIO_YAML_DIR}:")
        for flnm in os.listdir(dirs.SCENARIO_YAML_DIR):
            print(flnm)
        print(f"If you want to load from a different folder, change this to")
    if do_make:  # positional arguments
        make_scenario(args.yamls, dirs, args.name)
    if not (do_load or do_make or do_list):  # no positional - warn
        print("No yaml files provided. Nothing to do.")


if __name__ == "__main__":
    sys.exit(main())
