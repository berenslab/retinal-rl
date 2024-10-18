import os
import os.path as osp
import shutil
import struct
from glob import glob
from typing import Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from doom_creator.util.config import Config
from doom_creator.util.texture import TextureType

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


### Loading Datasets ###


def preload(
    type: TextureType,
    textures_dir: str,
    source_dir: Optional[str] = None,
    train: bool = True,
):
    if type.is_asset:
        assert source_dir is not None
        doomify = (
            type != TextureType.GABORS
        )  # only gabor images are not doomified somehow
        preload_assets(type, textures_dir, source_dir, doomify)
    else:
        preload_dataset(type, textures_dir, source_dir, train=train)


def preload_assets(
    asset_type: TextureType, textures_dir: str, assets_dir: str, doomify: bool = True
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
    dataset_type: TextureType,
    textures_dir: str,
    source_dir: Optional[str] = None,
    clean: Optional[bool] = None,
    train: bool = True,
):
    if clean is None:
        clean = source_dir is None

    out_path = osp.join(textures_dir, dataset_type.out_dir(not train))

    if source_dir is None:
        source_dir = out_path

    # check if resources/textures/$dataset$ exists
    if not osp.exists(out_path):
        dataset_wrapper = dataset_type.get_dataset_wrapper(source_dir, train)
        os.makedirs(out_path)

        # save images as pngs organized by word label
        for i in range(dataset_wrapper.num_classes):
            os.makedirs(osp.join(out_path, dataset_wrapper.label_to_str(i)))
        for i in range(len(dataset_wrapper.dataset)):
            png = osp.join(
                out_path,
                dataset_wrapper.label_to_str(dataset_wrapper.dataset[i][1]),
                str(i) + ".png",
            )  # TODO: instead of saving and then doomifying, doomify and save
            dataset_wrapper.dataset[i][0].save(png)
            doomify_image(png, 2)

        if clean:
            dataset_wrapper.clean(source_dir)


def check_preload(cfg: Config, test: bool):
    needed_types = set()
    for type_cfg in cfg.objects.values():
        for actor in type_cfg.actors.values():
            for i in range(len(actor.textures)):
                split_path = osp.split(actor.textures[i])
                t_type = split_path[-2]
                # assume second last part of path is the directory / texture type
                try:
                    t = TextureType(t_type)
                except Exception:
                    continue
                else:
                    needed_types.add(t)
                    if t.is_dataset and test:  # Update path if test set is wanted
                        actor.textures[i] = osp.join(
                            *split_path[:-2], t.out_dir(test), split_path[-1]
                        )
    return cfg, needed_types
