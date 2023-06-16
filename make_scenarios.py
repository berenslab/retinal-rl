import sys

from retinal_rl.scenarios.preload import preload_apples, preload_cifar10, preload_cifar100, preload_gabors, preload_mnist, preload_obstacles
from retinal_rl.scenarios.make import make_scenario2

def main():
    preload_apples()
    preload_obstacles()
    preload_gabors()
    preload_mnist()
    preload_cifar10()
    preload_cifar100()
    make_scenario2("gathering_apples")
    #make_scenario("gathering_gabors")
    #make_scenario("gathering_mnist")

if __name__ == '__main__':
    sys.exit(main())
