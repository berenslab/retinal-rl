from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpawnObjects:
    relative: bool = False
    range: float = 1000


@dataclass
class Metabolic:
    delay: int
    damage: int


@dataclass
class Actor:
    healing: Optional[int] = None
    damage: Optional[int] = None
    radius: Optional[int] = None
    textures: List[str] = field(default_factory=list)


@dataclass
class ObjectType:
    init: int
    delay: int
    actors: Dict[str, Actor]


@dataclass
class Objects:
    nourishment: ObjectType
    poison: ObjectType
    distractor: Optional[ObjectType] = None
    obstacle: Optional[ObjectType] = None


@dataclass
class Config:
    spawn_objects: SpawnObjects
    metabolic: Metabolic
    objects: Objects


from omegaconf import SCMode
from omegaconf.omegaconf import OmegaConf
import os.path as osp


### Load Config ###
def load_config(filenames: list[str], yaml_dir: str) -> Config:
    # list all config files
    file_pths = [osp.join(yaml_dir, "{0}.yaml".format(file)) for file in filenames]

    # Load all yaml files listed in flnms and combine into a single dictionary, recursively combining keys
    cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg, *[OmegaConf.load(f) for f in file_pths])
    # Convert to Config object
    return OmegaConf.to_container(cfg, structured_config_mode=SCMode.INSTANTIATE)
