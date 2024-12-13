from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT
from retinal_rl.models.objective import Objective
from runner.frameworks.classification.analyze import AnalysesCfg, analyze
from runner.frameworks.classification.dataset import get_datasets
from runner.frameworks.classification.initialize import initialize
from runner.frameworks.classification.train import train
from runner.frameworks.framework_interface import TrainingFramework


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

    def train(
        self,
        device: torch.device,
        brain: Brain,
        optimizer: torch.optim.Optimizer,
        objective: Optional[Objective[ContextT]] = None,
    ):
        # TODO: check objective type
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

    def analyze(
        self,
        device: torch.device,
        brain: Brain,
        objective: Optional[Objective[ContextT]] = None,
    ):
        cfg = AnalysesCfg(
            Path(self.cfg.path.run_dir),
            Path(self.cfg.path.plot_dir),
            Path(self.cfg.path.checkpoint_plot_dir),
            Path(self.cfg.path.data_dir),
            self.cfg.logging.use_wandb,
            self.cfg.logging.channel_analysis,
            self.cfg.logging.plot_sample_size,
        )
        analyze(
            cfg,
            device,
            brain,
            objective,
            self.histories,
            self.train_set,
            self.test_set,
            self.completed_epochs,
        )
