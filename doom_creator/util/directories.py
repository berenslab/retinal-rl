import os.path as osp
from dataclasses import dataclass
from typing import Optional


@dataclass
class Directories:
    cache_dir: str = "cache"
    resource_dir: str = osp.join("doom_creator", "resources")  # noqa: RUF009 as this is not really a dynamical call
    scenario_out_dir: Optional[str] = None
    build_dir: Optional[str] = None
    textures_dir: Optional[str] = None
    assets_dir: Optional[str] = None
    scenario_yaml_dir: Optional[str] = None
    dataset_dir: Optional[str] = None

    def __post_init__(self):
        self.CACHE_DIR = self.cache_dir
        self.SCENARIO_OUT_DIR = (
            osp.join(self.CACHE_DIR, "scenarios")
            if self.scenario_out_dir is None
            else self.scenario_out_dir
        )
        self.BUILD_DIR = (
            osp.join(self.SCENARIO_OUT_DIR, "build")
            if self.build_dir is None
            else self.build_dir
        )
        self.TEXTURES_DIR = (
            osp.join(self.CACHE_DIR, "textures")
            if self.textures_dir is None
            else self.textures_dir
        )
        self.RESOURCE_DIR = self.resource_dir
        self.ASSETS_DIR = (
            osp.join(self.RESOURCE_DIR, "assets")
            if self.assets_dir is None
            else self.assets_dir
        )
        self.SCENARIO_YAML_DIR = (
            osp.join(self.RESOURCE_DIR, "config")
            if self.scenario_yaml_dir is None
            else self.scenario_yaml_dir
        )
        self.DATASET_DIR = (
            self.TEXTURES_DIR if self.dataset_dir is None else self.dataset_dir
        )
