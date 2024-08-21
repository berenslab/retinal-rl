from enum import Enum
import os
import shutil
from typing import Optional

from num2words import num2words
import os.path as osp

from glob import glob
import struct
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from doom_creator.util import directories

### Util ###


# set offset for pngs based on zdooms grAb chunk, also optionally scale
def doomify_image(png, scale=1.0, shift=(0, 0), save_to=None):
    if save_to is None:
        save_to = png
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
    img.save(save_to, pnginfo=pnginfo)


class ImageDataType(Enum):
    APPLES = "apples"
    OBSTACLES = "obstacles"
    GABORS = "gabors"
    MNIST = "mnist"
    CIFAR10 = "cifar-10"
    CIFAR100 = "cifar-100"


### Loading Datasets ###


def preload(type: ImageDataType, textures_dir: str, source_dir: Optional[str] = None):
    if type in [ImageDataType.APPLES, ImageDataType.OBSTACLES, ImageDataType.GABORS]:
        assert source_dir is not None
        doomify = (
            type != ImageDataType.GABORS
        )  # only gabor images are not doomified somehow
        preload_assets(type, textures_dir, source_dir, doomify)
    else:
        preload_dataset(type, textures_dir, source_dir)


def preload_assets(
    asset_type: ImageDataType, textures_dir: str, assets_dir: str, doomify: bool = True
):
    type_str = asset_type.value
    assert osp.exists(osp.join(assets_dir, type_str))
    if not osp.exists(osp.join(textures_dir, type_str)):
        # copy images from resources
        shutil.copytree(
            osp.join(assets_dir, type_str), osp.join(textures_dir, type_str)
        )
        # Minor TODO: instead of copying adjust path when loading/saving for doomification
        # set offset for apple images
        if doomify:
            for png in glob(osp.join(textures_dir, type_str, "*.png")):
                doomify_image(png)


def preload_dataset(
    dataset_type: ImageDataType,
    textures_dir: str,
    data_path: Optional[str] = None,
    clean: Optional[bool] = None,
):
    if clean is None:
        clean = data_path is None
    if data_path is None:
        data_path = osp.join(textures_dir, dataset_type.value)

    # check if resources/textures/$dataset$ exists
    if not osp.exists(data_path):
        os.makedirs(data_path)

        if dataset_type is ImageDataType.MNIST:
            dataset = MNIST(data_path, download=True)
            label_to_str = num2words
            num_classes = len(dataset.classes)
            clean = clean_mnist(data_path)
        elif dataset_type is ImageDataType.CIFAR10:
            dataset = CIFAR10(data_path, download=True)
            label_to_str = lambda i: dataset.classes[i]
            num_classes = len(dataset.classes)
            clean = clean_cifar10
        elif dataset_type is ImageDataType.CIFAR100:
            dataset = CIFAR100(data_path, download=True)
            label_to_str = lambda i: dataset.classes[i]
            num_classes = len(dataset.classes)
            clean = clean_cifar100
        else:
            raise NotImplementedError(
                "Currently only mnist, cifar-10 and cifar-100 can be used"
            )

        # save images as pngs organized by word label
        for i in range(num_classes):
            os.makedirs(osp.join(data_path, label_to_str(i)))
        for i in range(len(dataset)):
            png = osp.join(
                data_path,
                label_to_str(dataset[i][1]),
                str(i) + ".png",
            )  # TODO: instead of saving and then doomifying, doomify and save
            dataset[i][0].save(png)
            doomify_image(png, 2)

        if clean:
            clean()


def clean_mnist(data_path):
    # remove all downloaded data except for the pngs
    shutil.rmtree(osp.join(data_path, "MNIST"), ignore_errors=True)


def clean_cifar10(data_path):
    os.remove(osp.join(data_path, "cifar-10-python.tar.gz"))
    shutil.rmtree(
        osp.join(data_path, "cifar-10-batches-py"),
        ignore_errors=True,
    )


def clean_cifar100(data_path: str):
    os.remove(osp.join(data_path, "cifar-100-python.tar.gz"))
    shutil.rmtree(
        osp.join(data_path, "cifar-100-python"),
        ignore_errors=True,
    )
