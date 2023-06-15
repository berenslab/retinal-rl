import sys

from retinal_rl.system.scenarios import make_scenario,load_apples

def main():
    #mnist_preload()
    #cifar10_preload()
    #cifar100_preload()
    load_apples()
    make_scenario()

if __name__ == '__main__':
    sys.exit(main())
