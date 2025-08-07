import argparse
import json
import os
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

# from retinal_rl.rl.sample_factory.observer import RetinalAlgoObserver
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from sample_factory.algo.learning.learner_factory import global_learner_factory
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.env_info import (
    obtain_env_info_in_a_separate_process,
)
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import (
    parse_full_cfg,
    parse_sf_args,
)
from sample_factory.enjoy import enjoy
from sample_factory.train import make_runner
from sample_factory.utils.typing import (
    Config,
    PolicyID,
)

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT
from retinal_rl.models.objective import Objective
from retinal_rl.rl.sample_factory.arguments import (
    add_retinal_env_args,
    add_retinal_env_eval_args,
    retinal_override_defaults,
)
from retinal_rl.rl.sample_factory.environment import register_retinal_env
from retinal_rl.rl.sample_factory.learner import RetinalLearner
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from retinal_rl.rl.sample_factory.observer import RetinalAlgoObserver
from runner.frameworks.classification.initialize import initialize
from runner.frameworks.framework_interface import TrainingFramework
from runner.util import create_brain


class SFFramework(TrainingFramework):
    def __init__(self, cfg: DictConfig, data_root: str):
        self.data_root = data_root

        # we need to convert to the sample_factory config style since we can not change the function signatures
        # of the library and that uses it _everywhere_
        self.sf_cfg = self.to_sf_cfg(cfg)

        # Register retinal environments and models.
        register_retinal_env(
            self.sf_cfg.env,
            self.data_root,
            self.sf_cfg.input_satiety,
            self.sf_cfg.allow_backwards,
            warp_exp=self.sf_cfg.warp_exp,
            warp_h=self.sf_cfg.warp_h,
            warp_w=self.sf_cfg.warp_w,
        )

        self.cfg = cfg

        # Validate brain configuration - not needed here, but useful to fail early
        env_info = obtain_env_info_in_a_separate_process(self.sf_cfg)
        SampleFactoryBrain.check_actor_critic(
            DictConfig(cfg.brain), env_info.action_space
        )

        global_model_factory().register_actor_critic_factory(SampleFactoryBrain)
        global_learner_factory().register_learner_factory(RetinalLearner)

    def initialize(self, brain: Brain, optimizer: torch.optim.Optimizer):
        # brain = SFFramework.load_brain_from_checkpoint(...)
        # TODO: Implement load brain and optimizer state
        # TODO: this initialize method is only used for wandb initialization...
        brain, optimizer, histories, epoch = initialize(self.cfg, brain, optimizer)
        return brain, optimizer

    def preset_model_and_optimizer(
        self, brain: Brain, optimizer: torch.optim.Optimizer, runner: Runner
    ):
        # Get information necessary for the learner from the actual runner
        env_info = runner.env_info
        policy_versions_tensor = runner.buffer_mgr.policy_versions
        param_server = runner.learners[0].param_server
        policy_id: PolicyID = param_server.policy_id

        # Init learner so we can use the default procedure to save the model and optimizer
        learner = RetinalLearner(
            self.sf_cfg, env_info, policy_versions_tensor, policy_id, param_server
        )
        learner.init()

        # Update the learners brain and optimizer and save them
        learner.actor_critic.brain = brain
        # learner.optimizer = optimizer #TODO: make sure optimizer is loadable
        learner.save()

    def train(
        self,
        device: torch.device,
        brain: Brain,
        optimizer: torch.optim.Optimizer,
        objective: Optional[Objective[ContextT]] = None,
    ):
        warnings.warn(
            "device, brain, optimizer and objective are initialized differently in sample_factory and thus their current state will be ignored"
        )
        # Run simulation
        if not (self.sf_cfg.dry_run):
            # HACK: Create the directory for the vizdoom logs, before vizdoom tries to do it:
            # Else envs will interfere in the creation process and crash, causing RolloutWorkers
            # to stop and thus the whole training to abort.
            (Path(self.cfg.path.run_dir) / "_vizdoom").mkdir(
                parents=True, exist_ok=True
            )

            cfg, runner = make_runner(self.sf_cfg)
            # if cfg.online_analysis:
            runner.register_observer(RetinalAlgoObserver(self.sf_cfg))

            status = runner.init()

            # update brain weights
            self.preset_model_and_optimizer(brain, optimizer, runner)

            if status == ExperimentStatus.SUCCESS:
                status = runner.run()
            print(status)

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

        if hasattr(cfg.dataset, "warp_exp") and cfg.dataset.warp_exp is not None:
            SFFramework._set_cfg_cli_argument(
                sf_cfg, "warp_exp", cfg.dataset.warp_exp
            )
            SFFramework._set_cfg_cli_argument(
                sf_cfg, "warp_h", cfg.dataset.vision_height
            )
            SFFramework._set_cfg_cli_argument(
                sf_cfg, "warp_w", cfg.dataset.vision_width
            )
        else:
            SFFramework._set_cfg_cli_argument(sf_cfg, "res_h", cfg.dataset.vision_height)
            SFFramework._set_cfg_cli_argument(sf_cfg, "res_w", cfg.dataset.vision_width)
        SFFramework._set_cfg_cli_argument(sf_cfg, "env", cfg.dataset.env_name)
        SFFramework._set_cfg_cli_argument(
            sf_cfg, "input_satiety", cfg.dataset.input_satiety
        )

        if hasattr(cfg.dataset, "allow_backwards"):
            SFFramework._set_cfg_cli_argument(
                sf_cfg, "allow_backwards", cfg.dataset.allow_backwards
            )
            # TODO: Doesn't need to be part of sf_cfg!
        else:
            SFFramework._set_cfg_cli_argument(sf_cfg, "allow_backwards", True)
            # TODO: move to default!

        if hasattr(cfg.dataset, "transforms"):
            SFFramework._set_cfg_cli_argument(
                sf_cfg, "transforms", OmegaConf.to_object(cfg.dataset.transforms)
            )
        else:
            SFFramework._set_cfg_cli_argument(sf_cfg, "transforms", [])
            # TODO: move to default!

        SFFramework._set_cfg_cli_argument(sf_cfg, "device", cfg.system.device)
        optimizer_name = str.lower(
            str.split(cfg.optimizer.optimizer._target_, sep=".")[-1]
        )
        SFFramework._set_cfg_cli_argument(sf_cfg, "optimizer", optimizer_name)

        SFFramework._set_cfg_cli_argument(
            sf_cfg, "brain", OmegaConf.to_object(cfg.brain)
        )
        SFFramework._set_cfg_cli_argument(
            sf_cfg, "objective", OmegaConf.to_object(cfg.optimizer.objective)
        )
        SFFramework._set_cfg_cli_argument(
            sf_cfg, "train_dir", os.path.join(cfg.path.run_dir, "train_dir")
        )

        # Set dirs needed for Analysis
        SFFramework._set_cfg_cli_argument(sf_cfg, "run_dir", cfg.path.run_dir)
        SFFramework._set_cfg_cli_argument(sf_cfg, "plot_dir", cfg.path.plot_dir)
        SFFramework._set_cfg_cli_argument(
            sf_cfg, "checkpoint_plot_dir", cfg.path.checkpoint_plot_dir
        )
        SFFramework._set_cfg_cli_argument(sf_cfg, "data_dir", cfg.path.data_dir)

        SFFramework._set_cfg_cli_argument(sf_cfg, "with_wandb", cfg.logging.use_wandb)
        SFFramework._set_cfg_cli_argument(sf_cfg, "wandb_dir", cfg.path.wandb_dir)
        if hasattr(
            cfg.brain.circuits, "rnn"
        ):  # TODO: remove samplefactory dependency for rnn setup
            SFFramework._set_cfg_cli_argument(sf_cfg, "use_rnn", True)
            # needed for initalizing the state correctly
            SFFramework._set_cfg_cli_argument(
                sf_cfg, "rnn_size", cfg.brain.circuits.rnn.rnn_size
            )
            SFFramework._set_cfg_cli_argument(
                sf_cfg, "rnn_num_layers", cfg.brain.circuits.rnn.rnn_num_layers
            )

        if hasattr(cfg, "samplefactory"):
            for attr in cfg.samplefactory:
                SFFramework._set_cfg_cli_argument(sf_cfg, attr, cfg.samplefactory[attr])
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
        SFFramework._set_cfg_cli_argument(self.sf_cfg, "save_video", True)
        SFFramework._set_cfg_cli_argument(self.sf_cfg, "no_render", True)
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


def brain_from_actor_critic(actor_critic: SampleFactoryBrain) -> Brain:
    return actor_critic.get_brain()  # TODO: Check if needed
