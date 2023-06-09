import os
import shutil
import subprocess

from num2words import num2words
import os.path as osp

import omg

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100



### Loading Datasets ###


def mnist_preload():
    # check if scenarios/resources/textures/mnist exists
    if not osp.exists("scenarios/resources/textures/mnist"):
    # if not, create it
        os.makedirs("scenarios/resources/textures/mnist")
    # download mnist images
        mnist = MNIST("scenarios/resources/textures/mnist", download=True)
    # save mnist images as pngs organized by word label
        for i in range(10):
            os.makedirs("scenarios/resources/textures/mnist/" + num2words(i))
        for i in range(len(mnist)):
            mnist[i][0].save("scenarios/resources/textures/mnist/" + num2words(mnist[i][1]) + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        shutil.rmtree("scenarios/resources/textures/mnist/MNIST", ignore_errors=True)

    else:
        print("mnist dir exists, files not downloaded)")

def cifar10_preload():
    # check if scenarios/resources/textures/cifar-10 exists
    if not osp.exists("scenarios/resources/textures/cifar-10"):
        # if not, create it
        os.makedirs("scenarios/resources/textures/cifar-10")
        # download cifar images
        cifar = CIFAR10("scenarios/resources/textures/cifar-10", download=True)
        # save cifar images as pngs organized by label name
        for i in range(10):
            os.makedirs("scenarios/resources/textures/cifar-10/" + cifar.classes[i])
        for i in range(len(cifar)):
            cifar[i][0].save("scenarios/resources/textures/cifar-10/" + cifar.classes[cifar[i][1]] + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        os.remove("scenarios/resources/textures/cifar-10/cifar-10-python.tar.gz")
        shutil.rmtree("scenarios/resources/textures/cifar-10/cifar-10-batches-py", ignore_errors=True)
    else:
        print("cifar-10 dir exists, files not downloaded")

def cifar100_preload():
    # check if scenarios/resources/textures/cifar-100 exists
    if not osp.exists("scenarios/resources/textures/cifar-100"):
        # if not, create it
        os.makedirs("scenarios/resources/textures/cifar-100")
        # download cifar images
        cifar = CIFAR100("scenarios/resources/textures/cifar-100", download=True)
        # save cifar images as pngs organized by label name
        for i in range(100):
            os.makedirs("scenarios/resources/textures/cifar-100/" + cifar.classes[i])
        for i in range(len(cifar)):
            cifar[i][0].save("scenarios/resources/textures/cifar-100/" + cifar.classes[cifar[i][1]] + "/" + str(i) + ".png")

        # remove all downloaded data except for the pngs
        os.remove("scenarios/resources/textures/cifar-100/cifar-100-python.tar.gz")
        shutil.rmtree("scenarios/resources/textures/cifar-100/cifar-100-python", ignore_errors=True)
    else:
        print("cifar-100 dir exists, files not downloaded")



### Creating Scenarios ###


def create_base_wad():

    bpth = "scenarios/resources/base"
    strct = omg.defstruct
    strct = [[omg.MarkerGroup, 'libraries', omg.Lump, 'A']] + strct
    wad = omg.WAD(structure=strct)

    print(wad.structure)
    grspth = osp.join(bpth,"grass.png")
    wndpth = osp.join(bpth,"wind.png")

    txtpth = osp.join(bpth,"TEXTMAP.txt")
    mappth = osp.join(bpth,"MAPINFO.txt")

    # Flats
    wad.ztextures['GRASS'] = omg.Graphic(from_file=grspth)

    # Data
    wad.data['WIND'] = omg.Lump(from_file=wndpth)
    wad.data['MAPINFO'] = omg.Lump(from_file=mappth)

    # Map preparation
    mpgrp = omg.LumpGroup()
    mpgrp['TEXTMAP'] = omg.Lump(from_file=txtpth)

    return wad,mpgrp

def make_scenario(task="gathering",texture="apples"):

    # take first 8 letters of task and capitalize
    lbonm = task[:8].upper()
    lbinm = lbonm[:5] + "SRC"
    scnnm = task + "_" + texture

    wad,mpgrp = create_base_wad()

    rscpth = "scenarios/resources"
    tskpth = osp.join(rscpth,"tasks")
    txtpth = osp.join(rscpth,"textures",texture)

    rappth = osp.join(txtpth,"red_apple.png")
    bappth = osp.join(txtpth,"blue_apple.png")
    decpth = osp.join(txtpth,"apples.decorate")

    bhipth = osp.join(rscpth,scnnm + ".acs")
    bhopth = osp.join(rscpth,scnnm + ".o")

    lbipth = osp.join(tskpth,task + ".acs")
    lbopth = osp.join(tskpth,task + ".o")

    # Compile ACS
    subprocess.call(["acc", "-i","/usr/share/acc", bhipth, bhopth])
    subprocess.call(["acc", "-i","/usr/share/acc", lbipth, lbopth])

    # Sprites
    wad.sprites['RAPPA0'] = omg.Graphic(from_file=rappth)
    wad.sprites['BAPPA0'] = omg.Graphic(from_file=bappth)

    # Decorate
    wad.data['DECORATE'] = omg.Lump(from_file=decpth)

    # Libraries
    wad.libraries[lbinm] = omg.Lump(from_file=lbipth)
    wad.libraries[lbonm] = omg.Lump(from_file=lbopth)

    # Map
    mpgrp['SCRIPTS'] = omg.Lump(from_file=bhipth)
    mpgrp['BEHAVIOR'] = omg.Lump(from_file=bhopth)
    wad.udmfmaps["MAP01"] = omg.UMapEditor(mpgrp).to_lumps()

    # Cleanup
    os.remove(bhopth)
    os.remove(lbopth)

    # Save to file
    wio = omg.WadIO(osp.join("scenarios",scnnm + ".wad"))
    write_order = omg.write_order
    write_order = ["libraries"] + write_order

    for group in write_order:
        wad.__dict__[group].save_wadio(wio, use_free=False)
    wio.save()
