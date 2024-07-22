from typing import Protocol
from retinal_rl.models.neural_circuit import NeuralCircuit
from retinal_rl.models.brain import Brain

from torch import optim
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple
from omegaconf import DictConfig
import torch


class RLEngine(Protocol):
    # TODO: Check if all parameters applicable and sort arguments
    # Potentially rename interface to even more abstract type if nothing RL specific needs to be handled here
    def initialize(cfg: DictConfig, brain: Brain, optimizer: optim.Optimizer): ...

    def train(
        self,
        cfg: DictConfig,
        brain: Brain,
        optimizer: optim.Optimizer,
        train_set: Dataset[Tuple[Tensor, int]],
        test_set: Dataset[Tuple[Tensor, int]],
        completed_epochs: int,
        histories: Dict[str, List[float]],
    ): ...

    def analysis(
        cfg: DictConfig,
        device: torch.device,
        brain: Brain,
        histories: Dict[str, List[float]],
        train_set: Dataset[Tuple[Tensor, int]],
        test_set: Dataset[Tuple[Tensor, int]],
        epoch: int,
        copy_checkpoint: bool = False,
    ): ...


class BrainInterface(Protocol):
    def get_brain() -> Brain: ...
