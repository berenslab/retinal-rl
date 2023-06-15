import os
import shutil
import subprocess

from num2words import num2words
from glob import glob
import os.path as osp

import omg

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100


### Loading Datasets ###


def load_apples():
    # check if scenarios/resources/textures/apples exists
    if not osp.exists("scenarios/resources/textures/apples"):
        # if not, create it
        os.makedirs("scenarios/resources/textures/apples")
        # download apple images
        rappth = "scenarios/resources/base/apples/red_apple.png"
        bappth = "scenarios/resources/base/apples/blue_apple.png"

        # copy red apple to scenarios/resources/textures/apples/nourishment-{1-5} and blue apples to poisson-{1-5}
        for i in range(1,6):
            os.makedirs("scenarios/resources/textures/apples/Nourishment_" + str(i))
            shutil.copyfile(rappth, "scenarios/resources/textures/apples/Nourishment_" + str(i) + "/apple.png")
            os.makedirs("scenarios/resources/textures/apples/Poison_" + str(i))
            shutil.copyfile(bappth, "scenarios/resources/textures/apples/Poison_" + str(i) + "/apple.png")

def load_mnist():
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

def load_cifar10():
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

def load_cifar100():
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
    wad = omg.WAD()

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


### Building Decorate Files ###

def actor_code(i,j):
    # Convert k to caps alpha string
    return chr(65 + i) + texture_code(j)

def texture_code(j):
    # Convert k to a 3-digit alpha all-caps string
    return chr(65 + j // 26 ** 2) + chr(65 + (j // 26) % 26) + chr(65 + j % 26)

def decorate_actor(actor_idx,actor_name,num_textures):

    # Multiline string for beginning of decorate file
    decorate = """ACTOR {0} : Inventory
    {{
        +INVENTORY.ALWAYSPICKUP
        States
            {{""".format(actor_name)

    for j in range(num_textures):
        decorate += """
            Tex{0}:
                {0} A -1""".format(actor_code(actor_idx,j))

    decorate += """
            }
    }\n\n"""

    return decorate

### Texture Packs ###

def load_textures(wad,actor_idx,pngs):

    num_textures = len(pngs)

    for j,png in enumerate(pngs):
        code = actor_code(actor_idx,j) + "A0"
        wad.sprites[code] = omg.Graphic(from_file=png)

    return num_textures

def make_scenario(task="gathering",texture="apples"):

    # Scenario name
    scnnm = task + "_" + texture

    txtdr = osp.join("scenarios/resources/textures",texture)
    # Get actor_names from base directory names in texture directory
    actor_names = [d for d in os.listdir(txtdr) if osp.isdir(osp.join(txtdr,d))]

    pngss = []
    # Get all pngs in each actor directory
    for actor_name in actor_names:
        actdr = osp.join(txtdr,actor_name)
        pngs = [osp.join(actdr,f) for f in os.listdir(actdr) if osp.isfile(osp.join(actdr,f)) and f.endswith(".png")]
        pngss.append(pngs)

    # Library path names
    lbonm = task[:8].upper()
    lbinm = lbonm[:5] + "SRC"

    wad,mpgrp = create_base_wad()

    decorate = ""

    for actor_idx,actor_name in enumerate(actor_names):
        num_textures = load_textures(wad,actor_idx,pngss[actor_idx])
        decorate += decorate_actor(actor_idx,actor_name,num_textures)

    # Decorate
    wad.data['DECORATE'] = omg.Lump(data=decorate.encode('utf-8'))

    rscpth = "scenarios/resources"
    tskpth = osp.join(rscpth,"tasks")

    bhipth = osp.join(rscpth,scnnm + ".acs")
    bhopth = osp.join(rscpth,scnnm + ".o")

    lbipth = osp.join(tskpth,task + ".acs")
    lbopth = osp.join(tskpth,task + ".o")

    # Compile ACS
    subprocess.call(["acc", "-i","/usr/share/acc", bhipth, bhopth])
    subprocess.call(["acc", "-i","/usr/share/acc", lbipth, lbopth])

    # Libraries
    wad.data[lbinm] = omg.Lump(from_file=lbipth)
    wad.libraries[lbonm] = omg.Lump(from_file=lbopth)

    # Map
    mpgrp['SCRIPTS'] = omg.Lump(from_file=bhipth)
    mpgrp['BEHAVIOR'] = omg.Lump(from_file=bhopth)
    wad.udmfmaps["MAP01"] = omg.UMapEditor(mpgrp).to_lumps()

    # Cleanup
    os.remove(bhopth)
    os.remove(lbopth)

    # Save to file
    wad.to_file(osp.join("scenarios",task + "_" + texture + ".wad"))

def make_gathering_apples():

    task="gathering"
    texture="apples"
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
    wad.to_file(osp.join("scenarios",task + "_" + texture + ".wad"))
