from dataclasses import dataclass
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import (
    parse_full_cfg,
    parse_sf_args,
    load_from_checkpoint,
)
from sample_factory.train import make_runner
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from retinal_rl.rl.analysis.simulation import get_checkpoint, get_brain_env
from torch import Tensor, optim
import argparse
from torch.utils.data import Dataset

from retinal_rl.models.brain import Brain
from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from retinal_rl.rl.sample_factory.arguments import (
    add_retinal_env_args,
    add_retinal_env_eval_args,
    retinal_override_defaults,
)
import json
from retinal_rl.rl.sample_factory.environment import register_retinal_env
from retinal_rl.rl.sample_factory.observer import RetinalAlgoObserver
import torch
from sample_factory.enjoy import enjoy

import os
from argparse import Namespace
from omegaconf.omegaconf import OmegaConf


class SFFramework(TrainingFramework):

    def __init__(self, cfg: DictConfig, data_root: str):
        self.data_root = data_root

        # we need to convert to the sample_factory config style since we can not change the function signatures
        # of the library and that uses it _everywhere_
        self.sf_cfg = self.to_sf_cfg(cfg)

        # Register retinal environments and models.
        register_retinal_env(self.sf_cfg.env, self.data_root, self.sf_cfg.input_satiety)
        global_model_factory().register_actor_critic_factory(SampleFactoryBrain)

    def train(self):
        # Run simulation
        if not (self.sf_cfg.dry_run):
            cfg, runner = make_runner(self.sf_cfg)
            if cfg.online_analysis:
                runner.register_observer(RetinalAlgoObserver(self.sf_cfg))

            status = runner.init()
            if status == ExperimentStatus.SUCCESS:
                status = runner.run()
            return status

    @staticmethod
    def load_brain_from_checkpoint(path: str, load_weights=True, device=None) -> Brain:
        with open(os.path.join(path, "config.json")) as f:
            config = Namespace(**json.load(f))
        checkpoint_dict, config = get_checkpoint(config)
        model_dict = checkpoint_dict["model"]
        brain_dict = {}
        for key in model_dict.keys():
            if "brain" in key:
                brain_dict[key[6:]] = model_dict[key]
        brain = Brain(**config["brain"])
        if load_weights:
            brain.load_state_dict(brain_dict)
        brain.to(device)
        return brain

    @staticmethod
    def load_brain_and_config(
        config_path: str, weights_path: str, device=None
    ) -> Brain:
        with open(os.path.join(config_path, "config.json")) as f:
            config = json.load(f)
        checkpoint_dict = torch.load(weights_path)
        model_dict = checkpoint_dict["model"]
        brain_dict = {}
        for key in model_dict.keys():
            if "brain" in key:
                brain_dict[key[6:]] = model_dict[key]
        brain = Brain(**config["brain"])
        brain.load_state_dict(brain_dict)
        brain.to(device)
        return brain

    def to_sf_cfg(self, cfg: DictConfig) -> Config:
        sf_cfg = self._get_default_cfg(cfg.rl.env_name)  # Load Defaults

        # overwrite default values with those set in cfg
        # TODO: which other parameters need to be set_
        self._set_cfg_cli_argument(sf_cfg, "learning_rate", cfg.training.learning_rate)
        # Using this function is necessary to make sure that the parameters are not overwritten when sample_factory loads a checkpoint

        self._set_cfg_cli_argument(sf_cfg, "res_h", cfg.rl.viewport_height)
        self._set_cfg_cli_argument(sf_cfg, "res_w", cfg.rl.viewport_width)
        self._set_cfg_cli_argument(sf_cfg, "env", cfg.rl.env_name)
        self._set_cfg_cli_argument(sf_cfg, "input_satiety", cfg.rl.input_satiety)
        self._set_cfg_cli_argument(sf_cfg, "device", cfg.system.device)
        self._set_cfg_cli_argument(sf_cfg, "optimizer", cfg.training.optimizer)

        self._set_cfg_cli_argument(sf_cfg, "brain", OmegaConf.to_object(cfg.brain))
        return sf_cfg

    def analyze(
        self,
        cfg: DictConfig,
        device: torch.device,
        brain: Brain,
        histories: Dict[str, List[float]],
        train_set: Dataset[Tuple[Tensor | int]],
        test_set: Dataset[Tuple[Tensor | int]],
        epoch: int,
        copy_checkpoint: bool = False,
    ):

        status = enjoy(self.sf_cfg)
        return status

    @staticmethod
    def _set_cfg_cli_argument(cfg: Namespace, name, value):
        """
        sample_factory overwrites arguments with those read from a checkpoint
        if they are not additionally added to the "cli_args"
        """
        cfg.__setattr__(name, value)
        cfg.cli_args[name] = value

    @staticmethod
    def _get_default_cfg(envname: str = "") -> argparse.Namespace:
        # TODO: get rid of intermediate parser step?!

        mock_argv = ["--env", envname]
        # SF needs an env name in argv.
        # Also, when loading from a checkpoint arguments in argv will not be overridden by arguments defined in the ckpt cfg.
        parser, cfg = parse_sf_args(mock_argv, evaluation=True)

        add_retinal_env_args(parser)
        # TODO: Replace with hydra style default to have all in one place & style (sf_config_hydra.yaml?)
        add_retinal_env_eval_args(parser)
        # Actually, discuss that. Would avoid having a unified interface
        retinal_override_defaults(parser)

        sf_cfg = parse_full_cfg(parser, mock_argv)
        return sf_cfg


def brain_from_actor_critic(actor_critic: SampleFactoryBrain) -> Brain:
    return actor_critic.get_brain()  # TODO: Check if needed
