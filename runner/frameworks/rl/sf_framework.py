import argparse
import json
import os
import warnings
from argparse import Namespace
from typing import Any, Dict, Optional

# from retinal_rl.rl.sample_factory.observer import RetinalAlgoObserver
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import (
    load_from_checkpoint,
    parse_full_cfg,
    parse_sf_args,
)
from sample_factory.enjoy import enjoy
from sample_factory.train import make_runner
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT
from retinal_rl.models.objective import Objective
from retinal_rl.rl.sample_factory.arguments import (
    add_retinal_env_args,
    add_retinal_env_eval_args,
    retinal_override_defaults,
)
from retinal_rl.rl.sample_factory.environment import register_retinal_env
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from runner.frameworks.framework_interface import TrainingFramework
from runner.util import create_brain


class SFFramework(TrainingFramework):
    def __init__(self, cfg: DictConfig, data_root: str):
        self.data_root = data_root

        # we need to convert to the sample_factory config style since we can not change the function signatures
        # of the library and that uses it _everywhere_
        self.sf_cfg = self.to_sf_cfg(cfg)

        # Register retinal environments and models.
        register_retinal_env(self.sf_cfg.env, self.data_root, self.sf_cfg.input_satiety)
        global_model_factory().register_actor_critic_factory(SampleFactoryBrain)

    def initialize(self, brain: Brain, optimizer: torch.optim.Optimizer):
        # brain = SFFramework.load_brain_from_checkpoint(...)
        # TODO: Implement load brain and optimizer state
        return brain, optimizer

    def train(
        self,
        device: torch.device,
        brain: Brain,
        optimizer: torch.optim.Optimizer,
        objective: Optional[Objective[ContextT]] = None,
    ):
        warnings.warn(
            "device, brain, optimizer are initialized differently in sample_factory and thus their current state will be ignored"
        )
        warnings.warn(
            "objective is currently not supported for sample factory simulations"
        )
        # Run simulation
        if not (self.sf_cfg.dry_run):
            cfg, runner = make_runner(self.sf_cfg)
            # if cfg.online_analysis:
            #     runner.register_observer(RetinalAlgoObserver(self.sf_cfg))

            status = runner.init()
            if status == ExperimentStatus.SUCCESS:
                status = runner.run()
            print(status)

    @staticmethod
    def load_brain_from_checkpoint(
        path: str, load_weights: bool = True, device: Optional[torch.device] = None
    ) -> Brain:
        with open(os.path.join(path, "config.json")) as f:
            config = Namespace(**json.load(f))
        checkpoint_dict, config = SFFramework.get_checkpoint(config)
        config = DictConfig(config)
        model_dict: Dict[str, Any] = checkpoint_dict["model"]
        brain_dict: Dict[str, Any] = {}
        for key in model_dict:
            if "brain" in key:
                brain_dict[key[6:]] = model_dict[key]
        brain = create_brain(config.brain)
        if load_weights:
            brain.load_state_dict(brain_dict)
        brain.to(device)
        return brain

    @staticmethod
    def load_brain_and_config(
        config_path: str, weights_path: str, device: Optional[torch.device] = None
    ) -> Brain:
        with open(os.path.join(config_path, "config.json")) as f:
            config = DictConfig(json.load(f))
        checkpoint_dict = torch.load(weights_path)
        model_dict = checkpoint_dict["model"]
        brain_dict = {}
        for key in model_dict:
            if "brain" in key:
                brain_dict[key[6:]] = model_dict[key]
        brain = create_brain(config.brain)
        brain.load_state_dict(brain_dict)
        brain.to(device)
        return brain

    @staticmethod
    def to_sf_cfg(cfg: DictConfig) -> Config:
        sf_cfg = SFFramework._get_default_cfg(cfg.dataset.env_name)  # Load Defaults

        # overwrite default values with those set in cfg
        # TODO: which other parameters need to be set_
        SFFramework._set_cfg_cli_argument(
            sf_cfg, "learning_rate", cfg.optimizer.optimizer.lr
        )
        # Using this function is necessary to make sure that the parameters are not overwritten when sample_factory loads a checkpoint

        SFFramework._set_cfg_cli_argument(sf_cfg, "res_h", cfg.dataset.vision_width)
        SFFramework._set_cfg_cli_argument(sf_cfg, "res_w", cfg.dataset.vision_height)
        SFFramework._set_cfg_cli_argument(sf_cfg, "env", cfg.dataset.env_name)
        SFFramework._set_cfg_cli_argument(
            sf_cfg, "input_satiety", cfg.dataset.input_satiety
        )
        SFFramework._set_cfg_cli_argument(sf_cfg, "device", cfg.system.device)
        optimizer_name = str.lower(
            str.split(cfg.optimizer.optimizer._target_, sep=".")[-1]
        )
        SFFramework._set_cfg_cli_argument(sf_cfg, "optimizer", optimizer_name)

        SFFramework._set_cfg_cli_argument(
            sf_cfg, "brain", OmegaConf.to_object(cfg.brain)
        )
        SFFramework._set_cfg_cli_argument(
            sf_cfg, "train_dir", os.path.join(cfg.path.run_dir, "train_dir")
        )
        SFFramework._set_cfg_cli_argument(sf_cfg, "with_wandb", cfg.logging.use_wandb)
        SFFramework._set_cfg_cli_argument(sf_cfg, "wandb_dir", cfg.path.wandb_dir)
        return sf_cfg

    def analyze(
        self,
        device: torch.device,
        brain: Brain,
        objective: Optional[Objective[ContextT]] = None,
    ):
        warnings.warn(
            "device, brain, optimizer are initialized differently in sample_factory and thus their current state will be ignored"
        )
        enjoy(self.sf_cfg)
        # TODO: Implement analyze function for sf framework

    @staticmethod
    def _set_cfg_cli_argument(cfg: Namespace, name: str, value: Any):
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
        parser, _ = parse_sf_args(mock_argv, evaluation=True)

        add_retinal_env_args(parser)
        # TODO: Replace with hydra style default to have all in one place & style (sf_config_hydra.yaml?)
        add_retinal_env_eval_args(parser)
        # Actually, discuss that. Would avoid having a unified interface
        retinal_override_defaults(parser)

        return parse_full_cfg(parser, mock_argv)

    @staticmethod
    def get_checkpoint(cfg: Config) -> tuple[Dict[str, Any], AttrDict]:
        """
        Load the model from checkpoint, initialize the environment, and return both.
        """
        # verbose = False

        cfg = load_from_checkpoint(cfg)

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

        policy_id = cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*"
        )
        checkpoint_dict: Dict[str, Any] = Learner.load_checkpoint(checkpoints, device)

        return checkpoint_dict, cfg


def brain_from_actor_critic(actor_critic: SampleFactoryBrain) -> Brain:
    return actor_critic.get_brain()  # TODO: Check if needed
