from typing import Optional
import torch
from omegaconf import DictConfig

from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT
from retinal_rl.models.objective import Objective
from runner.classification.analyze import analyze
from runner.classification.dataset import get_datasets
from runner.classification.initialize import initialize
from runner.classification.train import train


class ClassificationFramework(TrainingFramework):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_set, self.test_set = get_datasets(self.cfg)

    def initialize(self, brain: Brain, optimizer: torch.optim.Optimizer):
        brain, optimizer, self.histories, self.completed_epochs = initialize(
            self.cfg,
            brain,
            optimizer,
        )
        return brain, optimizer

    def train(self, device: torch.device, brain: Brain, optimizer: torch.optim.Optimizer, objective: Optional[Objective[ContextT]] = None):
        #TODO: check objective type
        train(
                self.cfg,
                device,
                brain,
                objective,
                optimizer,
                self.train_set,
                self.test_set,
                self.completed_epochs,
                self.histories,
            )

    def analyze(self, device: torch.device, brain: Brain, objective: Optional[Objective[ContextT]] = None):
        analyze(
                self.cfg,
                device,
                brain,
                objective,
                self.histories,
                self.train_set,
                self.test_set,
                self.completed_epochs,
            )
