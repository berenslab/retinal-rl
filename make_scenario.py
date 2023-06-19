import sys

from retinal_rl.scenarios.preload import preload_apples, preload_cifar10, preload_cifar100, preload_gabors, preload_mnist, preload_obstacles
from retinal_rl.scenarios.make import make_scenario

def main():

    # Parse args
    argv = sys.argv[1:]
    # If argv include all
    if "all" in argv:
    
        preload_apples()
        preload_obstacles()
        preload_gabors()
        preload_mnist()
        preload_cifar10()
        preload_cifar100()
        make_scenario("gathering_apples")
        make_scenario("gathering_gabors")
        make_scenario("gathering_mnist")
        make_scenario("gathering_cifar10")
        make_scenario("obstructed_apples")
        make_scenario("obstructed_gabors")
        make_scenario("obstructed_mnist")
        make_scenario("obstructed_cifar10")

    # If argv is empty
    elif len(argv) == 0:
        print("Please specify a scenario to make.")

    else:
        for scnr in argv:
            make_scenario(scnr)

if __name__ == '__main__':
    sys.exit(main())
