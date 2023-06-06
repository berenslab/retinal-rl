import os
import shutil
import subprocess

from os.path import join as pjoin

import omg

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

def mnist_preload():
    # check if scenarios/resources/graphics/mnist exists
    if not os.path.exists("scenarios/resources/graphics/mnist"):
    # if not, create it
        os.makedirs("scenarios/resources/graphics/mnist")
    # download mnist images
        mnist = MNIST("scenarios/resources/graphics/mnist", download=True)
    # save mnist images as pngs organized by label
        for i in range(10):
            os.makedirs("scenarios/resources/graphics/mnist/" + str(i))
        for i in range(len(mnist)):
            mnist[i][0].save("scenarios/resources/graphics/mnist/" + str(mnist[i][1]) + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        shutil.rmtree("scenarios/resources/graphics/mnist/MNIST", ignore_errors=True)

    else:
        print("mnist dir exists, files not downloaded)")

def cifar10_preload():
    # check if scenarios/resources/graphics/cifar-10 exists
    if not os.path.exists("scenarios/resources/graphics/cifar-10"):
        # if not, create it
        os.makedirs("scenarios/resources/graphics/cifar-10")
        # download cifar images
        cifar = CIFAR10("scenarios/resources/graphics/cifar-10", download=True)
        # save cifar images as pngs organized by label name
        for i in range(10):
            os.makedirs("scenarios/resources/graphics/cifar-10/" + cifar.classes[i])
        for i in range(len(cifar)):
            cifar[i][0].save("scenarios/resources/graphics/cifar-10/" + cifar.classes[cifar[i][1]] + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        os.remove("scenarios/resources/graphics/cifar-10/cifar-10-python.tar.gz")
        shutil.rmtree("scenarios/resources/graphics/cifar-10/cifar-10-batches-py", ignore_errors=True)
    else:
        print("cifar-10 dir exists, files not downloaded")

def cifar100_preload():
    # check if scenarios/resources/graphics/cifar-100 exists
    if not os.path.exists("scenarios/resources/graphics/cifar-100"):
        # if not, create it
        os.makedirs("scenarios/resources/graphics/cifar-100")
        # download cifar images
        cifar = CIFAR100("scenarios/resources/graphics/cifar-100", download=True)
        # save cifar images as pngs organized by label name
        for i in range(100):
            os.makedirs("scenarios/resources/graphics/cifar-100/" + cifar.classes[i])
        for i in range(len(cifar)):
            cifar[i][0].save("scenarios/resources/graphics/cifar-100/" + cifar.classes[cifar[i][1]] + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        os.remove("scenarios/resources/graphics/cifar-100/cifar-100-python.tar.gz")
        shutil.rmtree("scenarios/resources/graphics/cifar-100/cifar-100-python", ignore_errors=True)
    else:
        print("cifar-100 dir exists, files not downloaded")

def create_base_wad(gpth="scenarios/resources/graphics",spth="scenarios/resources/scripts"):

    wad = omg.WAD()

    bgpth = pjoin(gpth, "base")

    grspth = pjoin(bgpth,"grass.png")
    wndpth = pjoin(bgpth,"wind.png")

    bspth = pjoin(spth, "base")

    txtpth = pjoin(bspth,"TEXTMAP.txt")
    mappth = pjoin(bspth,"MAPINFO.txt")

    # Flats
    wad.ztextures['GRASS'] = omg.Graphic(from_file=grspth)

    # Data
    wad.data['WIND'] = omg.Lump(from_file=wndpth)
    wad.data['MAPINFO'] = omg.Lump(from_file=mappth)

    # Map preparation
    lgrp = omg.LumpGroup()
    lgrp['TEXTMAP'] = omg.Lump(from_file=txtpth)

    return wad,lgrp

def create_gathering_apples(rpth="apples",gpth="scenarios/resources/graphics",spth="scenarios/resources/scripts"):

    wad,lgrp = create_base_wad(gpth,spth)

    rgpth = pjoin(gpth, rpth)

    rappth = pjoin(rgpth,"red_apple.png")
    bappth = pjoin(rgpth,"blue_apple.png")

    rspth = pjoin(spth, rpth)

    decpth = pjoin(rspth,"DECORATE.txt")
    scrpth = pjoin(rspth,"SCRIPTS.txt")
    behpth = pjoin(rspth,"SCRIPTS.o")


    # Sprites
    wad.sprites['RAPPA0'] = omg.Graphic(from_file=rappth)
    wad.sprites['BAPPA0'] = omg.Graphic(from_file=bappth)

    # Decorate
    wad.data['DECORATE'] = omg.Lump(from_file=decpth)

    # Compile ACS
    subprocess.call(["acc", "-i","/usr/share/acc", scrpth, behpth])

    # Map
    lgrp['SCRIPTS'] = omg.Lump(from_file=scrpth)
    lgrp['BEHAVIOR'] = omg.Lump(from_file=behpth)

    # Cleanup
    wad.udmfmaps["MAP01"] = omg.UMapEditor(lgrp).to_lumps()
    
    wad.to_file("scenarios/apples.wad")
    
    os.remove(behpth)
