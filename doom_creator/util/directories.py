from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class Directories:
    cache_dir: Path = "cache"
    resource_dir: Path = Path("doom_creator", "resources")
    scenario_out_dir: Optional[Path] = None
    build_dir: Optional[Path] = None
    textures_dir: Optional[Path] = None
    assets_dir: Optional[Path] = None
    scenario_yaml_dir: Optional[Path] = None
    dataset_dir: Optional[Path] = None

    def __post_init__(self):
        self.CACHE_DIR: Path = self.cache_dir
        self.SCENARIO_OUT_DIR: Path = (
            Path(self.CACHE_DIR, "scenarios")
            if self.scenario_out_dir is None
            else self.scenario_out_dir
        )
        self.BUILD_DIR: Path = (
            Path(self.SCENARIO_OUT_DIR, "build")
            if self.build_dir is None
            else self.build_dir
        )
        self.TEXTURES_DIR: Path = (
            Path(self.CACHE_DIR, "textures")
            if self.textures_dir is None
            else self.textures_dir
        )
        self.RESOURCE_DIR: Path = self.resource_dir
        self.ASSETS_DIR: Path = (
            Path(self.RESOURCE_DIR, "assets")
            if self.assets_dir is None
            else self.assets_dir
        )
        self.SCENARIO_YAML_DIR: Path = (
            Path(self.RESOURCE_DIR, "config")
            if self.scenario_yaml_dir is None
            else self.scenario_yaml_dir
        )
        self.DATASET_DIR: Path = (
            self.TEXTURES_DIR if self.dataset_dir is None else self.dataset_dir
        )
