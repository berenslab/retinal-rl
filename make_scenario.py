import sys
import argparse

from retinal_rl.scenarios.preload import preload_apples, preload_cifar10, preload_cifar100, preload_gabors, preload_mnist, preload_obstacles
from retinal_rl.scenarios.make import make_scenario


def make_parser():
    # Initialize parser
    parser = argparse.ArgumentParser(description="Make a scenario for retinal-rl")
    # Add option to run preload
    parser.add_argument("--preload", action="store_true", help="Preload resources")
    # Positional argument for scenario yaml files (required, can be multiple)
    parser.add_argument("yamls", nargs="*", help="Component yaml files for scenario")
    # Argument for optional scneario name
    parser.add_argument("--name", help="Name of scenario to make")
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

    # positional arguments
    if len(args.yamls) > 0:
        make_scenario(args.yamls, args.name)
    else:
        # no positional, warn and exit
        print("No yaml files provided.  Nothing to do.")

if __name__ == '__main__':
    sys.exit(main())
