import os.path as osp
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from omegaconf import SCMode
from omegaconf.omegaconf import OmegaConf


@dataclass
class SpawnObjects:
    relative: bool = False
    range: int = 1000


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
class ObjectTypeVars:
    init: int
    delay: int
    actors: Dict[str, Actor]


class ObjectType(Enum):
    nourishment = "nourishment"
    poison = "poison"
    distractor = "distractor"
    obstacle = "obstacle"


@dataclass
class Config:
    spawn_objects: SpawnObjects
    metabolic: Metabolic
    objects: Dict[ObjectType, ObjectTypeVars]


### Load Config ###
def load(filenames: list[str], yaml_dir: str) -> Config:
    # list all config files
    file_pths = [osp.join(yaml_dir, f"{file}.yaml") for file in filenames]

    # Load all yaml files listed in flnms and combine into a single dictionary, recursively combining keys
    cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg, *[OmegaConf.load(f) for f in file_pths])
    # Convert to Config object
    return OmegaConf.to_container(cfg, structured_config_mode=SCMode.INSTANTIATE)
