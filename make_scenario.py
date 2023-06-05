import os
import shutil
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

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
    print("mnist images already downloaded")

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
    print("cifar-10 images already downloaded")

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
    print("cifar-100 images already downloaded")

import os
import omg
import subprocess

wad = omg.WAD()
pth = "data/"

rappth = pth + "red_apple.png"
bappth = pth + "blue_apple.png"
grspth = pth + "grass.png"
wndpth = pth + "wind.png"
mappth = pth + "MAPINFO.txt"
decpth = pth + "DECORATE.txt"
scrpth = pth + "SCRIPTS.txt"
behpth = pth + "SCRIPTS.o"
txtpth = pth + "TEXTMAP.txt"

# Sprites
wad.sprites['RAPPA0'] = omg.Graphic(from_file=rappth)
wad.sprites['BAPPA0'] = omg.Graphic(from_file=bappth)

# Flats
wad.ztextures['GRASS'] = omg.Graphic(from_file=grspth)

# Data
wad.data['WIND'] = omg.Lump(from_file=wndpth)
wad.data['MAPINFO'] = omg.Lump(from_file=mappth)
wad.data['DECORATE'] = omg.Lump(from_file=decpth)

# Compile ACS
subprocess.call(["acc", "-i","/usr/share/acc", scrpth, behpth])

# Map
lgrp = omg.LumpGroup()
lgrp['SCRIPTS'] = omg.Lump(from_file=scrpth)
lgrp['BEHAVIOR'] = omg.Lump(from_file=behpth)
lgrp['TEXTMAP'] = omg.Lump(from_file=txtpth)

wad.udmfmaps["MAP01"] = omg.UMapEditor(lgrp).to_lumps()

wad.to_file("apples.wad")

os.remove(behpth)
