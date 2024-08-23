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