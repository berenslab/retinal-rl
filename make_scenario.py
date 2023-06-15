import sys

from retinal_rl.scenarios.preload import preload_apples
from retinal_rl.scenarios.make import make_scenario

def main():
    #mnist_preload()
    #cifar10_preload()
    #cifar100_preload()
    preload_apples()
    make_scenario("gathering_apples")

if __name__ == '__main__':
    sys.exit(main())
