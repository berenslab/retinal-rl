from typing import Optional, Protocol, Tuple

import torch

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT
from retinal_rl.models.objective import Objective


class TrainingFramework(Protocol):
    def initialize(
        self, brain: Brain, optimizer: torch.optim.Optimizer
    ) -> Tuple[Brain, torch.optim.Optimizer]: ...

    def train(
        self,
        device: torch.device,
        brain: Brain,
        optimizer: torch.optim.Optimizer,
        objective: Optional[Objective[ContextT]] = None,
    ): ...

    def analyze(
        self,
        device: torch.device,
        brain: Brain,
        objective: Optional[Objective[ContextT]] = None,
    ): ...
