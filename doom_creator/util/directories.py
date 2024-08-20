import os.path as osp

CACHE_DIR = "cache" #TODO: One should be able to set this
SCENARIO_OUT_DIR = osp.join(CACHE_DIR, "scenarios")

BUILD_DIR = osp.join(SCENARIO_OUT_DIR, "build")
TEXTURES_DIR = osp.join(CACHE_DIR, "textures")

RESOURCE_DIR = osp.join("doom_creator","resources")
ASSETS_DIR = osp.join(RESOURCE_DIR, "assets")
SCENARIO_YAML_DIR = osp.join(RESOURCE_DIR, "config")