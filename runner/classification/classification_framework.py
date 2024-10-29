from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.models.brain import Brain
from runner.classification.analyze import analyze
from runner.classification.dataset import get_datasets
from runner.classification.initialize import initialize
from runner.classification.train import train


class ClassificationFramework(TrainingFramework):

    def __init__(self, cfg: DictConfig, brain: Brain, optimizer: torch.optim.Optimizer):
        self.cfg = cfg
        self.brain, self.optimizer, self.histories, self.completed_epochs = initialize(
            cfg,
            brain,
            optimizer,
        )
        self.train_set, self.test_set = self.get_datasets(self.cfg)

    def get_datasets(self):
        return get_datasets(self.cfg)

    def train(self, device, objective, optimizer):
        train(
                self.cfg,
                device,
                self.brain,
                objective,
                optimizer,
                self.train_set,
                self.test_set,
                self.completed_epochs,
                self.histories,
            )

    def analyze(self, device, objective):
        analyze(
                self.cfg,
                device,
                self.brain,
                objective,
                self.histories,
                self.train_set,
                self.test_set,
                self.completed_epochs,
            )
