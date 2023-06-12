import sys

from retinal_rl.system.scenarios import make_scenario,mnist_preload,cifar10_preload,cifar100_preload

def main():
    #mnist_preload()
    #cifar10_preload()
    #cifar100_preload()
    make_scenario()

if __name__ == '__main__':
    sys.exit(main())
