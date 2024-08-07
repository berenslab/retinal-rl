import os
import shutil

from num2words import num2words
import os.path as osp

from glob import glob
import struct
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from retinal_rl.util import resources_dir


### Base Directories ###

assets_dir = osp.join(resources_dir, "scenario_assets")
cache_dir = "cache"
textures_dir = osp.join(cache_dir, "textures")


### Util ###


# set offset for pngs based on zdooms grAb chunk, also optionally scale
def doomify_image(png, scale=1.0, shift=(0, 0)):
    img = Image.open(png)
    if scale != 1.0:
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.Resampling.NEAREST,
        )
    # get width and height
    width, height = img.size
    width += shift[0]
    height += shift[1]
    pnginfo = PngInfo()
    pnginfo.add(b"grAb", struct.pack(">II", width // 2, height))
    img.save(png, pnginfo=pnginfo)


### Loading Datasets ###


def preload_apples():
    # check if textures_dir/apples exists
    if not osp.exists(osp.join(textures_dir, "apples")):
        # copy apple images from resources/base
        shutil.copytree(
            osp.join(assets_dir, "apples"), osp.join(textures_dir, "apples")
        )
        # set offset for apple images
        for png in glob(osp.join(textures_dir, "apples/*.png")):
            doomify_image(png)


def preload_obstacles():
    if not osp.exists(osp.join(textures_dir, "obstacles")):
        shutil.copytree(
            osp.join(assets_dir, "obstacles"), osp.join(textures_dir, "obstacles")
        )
        # set offset for obstacle images
        for png in glob(osp.join(textures_dir, "obstacles/*.png")):
            doomify_image(png)


def preload_gabors():
    if not osp.exists(osp.join(textures_dir, "gabors")):
        shutil.copytree(
            osp.join(assets_dir, "gabors"), osp.join(textures_dir, "gabors")
        )


def preload_mnist():
    # check if resources/textures/mnist exists
    if not osp.exists(osp.join(textures_dir, "mnist")):
        # if not, create it
        os.makedirs(osp.join(textures_dir, "mnist"))
        # download mnist images
        mnist = MNIST(osp.join(textures_dir, "mnist"), download=True)
        # save mnist images as pngs organized by word label
        for i in range(10):
            os.makedirs(osp.join(textures_dir, "mnist", num2words(i)))
        for i in range(len(mnist)):
            png = osp.join(
                textures_dir, "mnist", num2words(mnist[i][1]), str(i) + ".png"
            )
            mnist[i][0].save(png)
            doomify_image(png, 2)

        # remove all downloaded data except for the pngs
        shutil.rmtree(osp.join(textures_dir, "mnist", "MNIST"), ignore_errors=True)


def preload_cifar10():
    # check if resources/textures/cifar-10 exists
    if not osp.exists(osp.join(textures_dir, "cifar-10")):
        # if not, create it
        os.makedirs(osp.join(textures_dir, "cifar-10"))
        # download cifar images
        cifar = CIFAR10(osp.join(textures_dir, "cifar-10"), download=True)
        # save cifar images as pngs organized by label name
        for i in range(10):
            os.makedirs(osp.join(textures_dir, "cifar-10/", cifar.classes[i]))
        for i in range(len(cifar)):
            png = osp.join(
                textures_dir, "cifar-10", cifar.classes[cifar[i][1]], str(i) + ".png"
            )
            cifar[i][0].save(png)
            doomify_image(png, 2)
            # doomify_image(png,shift=(0,16))

        # remove all downloaded data except for the pngs
        os.remove(osp.join(textures_dir, "cifar-10/cifar-10-python.tar.gz"))
        shutil.rmtree(
            osp.join(textures_dir, "cifar-10/cifar-10-batches-py"), ignore_errors=True
        )


def preload_cifar100():
    # check if resources/textures/cifar-100 exists
    if not osp.exists(osp.join(textures_dir, "cifar-100")):
        # if not, create it
        os.makedirs(osp.join(textures_dir, "cifar-100"))
        # download cifar images
        cifar = CIFAR100(osp.join(textures_dir, "cifar-100"), download=True)
        # save cifar images as pngs organized by label name
        for i in range(100):
            os.makedirs(osp.join(textures_dir, "cifar-100", cifar.classes[i]))
        for i in range(len(cifar)):
            png = osp.join(
                textures_dir, "cifar-100", cifar.classes[cifar[i][1]], str(i) + ".png"
            )
            cifar[i][0].save(png)
            doomify_image(png, 2)

        # remove all downloaded data except for the pngs
        os.remove(osp.join(textures_dir, "cifar-100/cifar-100-python.tar.gz"))
        shutil.rmtree(
            osp.join(textures_dir, "cifar-100/cifar-100-python"), ignore_errors=True
        )
