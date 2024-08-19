from typing import Protocol
from retinal_rl.models.neural_circuit import NeuralCircuit
from retinal_rl.models.brain import Brain

from torch import optim
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple
from omegaconf import DictConfig
import torch


class TrainingFramework(Protocol):
    # TODO: Check if all parameters applicable and sort arguments
    # Especially get rid of config were possible (train? initialize could store all relevant parameters...)
    def train(self): ...

    # TODO: make static to be able to evaluate models from other stuff as well?
    def analyze(
        self,
        cfg: DictConfig,
        device: torch.device,
        brain: Brain,
        histories: Dict[str, List[float]],
        train_set: Dataset[Tuple[Tensor, int]],
        test_set: Dataset[Tuple[Tensor, int]],
        epoch: int,
        copy_checkpoint: bool = False,
    ): ...
