from dataclasses import dataclass
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args, load_from_checkpoint
from sample_factory.train import make_runner
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from retinal_rl.rl.analysis.simulation import get_checkpoint, get_brain_env
from torch import Tensor, optim
from torch.utils.data import Dataset

from retinal_rl.models.brain import Brain
from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.rl.sample_factory.config_defaults import SfDefaults
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from retinal_rl.rl.system.arguments import (add_retinal_env_args,
                                            add_retinal_env_eval_args,
                                            retinal_override_defaults)
import json
from retinal_rl.rl.system.environment import register_retinal_env
from retinal_rl.rl.system.exec import RetinalAlgoObserver
import warnings
import torch
from sample_factory.enjoy import enjoy

import os
from argparse import Namespace
from omegaconf.omegaconf import OmegaConf

def get_default_cfg(envname: str = "") -> Config: # TODO: get rid of intermediate parser step?!

    mock_argv = ["--env", envname]
    # SF needs an env name in argv.
    # Also, when loading from a checkpoint arguments in argv will not be overridden by arguments defined in the ckpt cfg. 
    parser, cfg = parse_sf_args(mock_argv, evaluation=True)


    add_retinal_env_args(parser) # TODO: Replace with hydra style default to have all in one place & style (sf_config_hydra.yaml?)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)

    sf_cfg = parse_full_cfg(parser, mock_argv)
    sf_cfg.cli_args = {} # We don't want cli_args as they will be used to overwrite other args in sample_factory!
    return sf_cfg

class SFFramework(TrainingFramework):

    # def __init__(self, cfg: DictConfig, brain: Brain, optimizer: optim.Optimizer, data_root: str):
    #     self.initialize(cfg, brain, optimizer, data_root)

    def train(
        self,
        cfg: DictConfig,
        device: torch.device,
        brain: Brain,
        optimizer: optim.Optimizer,
        train_set: Dataset[Tuple[Tensor, int]],
        test_set: Dataset[Tuple[Tensor, int]],
        completed_epochs: int,
        histories: Dict[str, List[float]],
    ):
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
    def load_brain_from_checkpoint(path: str, device=None) -> torch.nn.Module:
        with open(os.path.join(path, "config.json")) as f:
            config = Namespace(**json.load(f))
        checkpoint_dict,config = get_checkpoint(config)
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
        sf_cfg = get_default_cfg()  # Load Defaults

        # overwrite default values with those set in cfg
        # TODO: merge cfg and sf_cfg
        sf_cfg.learning_rate = cfg.training.learning_rate

        sf_cfg.res_h = cfg.rl.viewport_height
        sf_cfg.res_w = cfg.rl.viewport_width
        sf_cfg.env = cfg.rl.env_name
        sf_cfg.input_satiety = cfg.rl.input_satiety
        
        sf_cfg.brain = OmegaConf.to_object(cfg.brain)
        return sf_cfg
    
    def initialize(self, cfg: DictConfig, brain: Brain, optimizer: optim.Optimizer, data_root: str):
        self.data_root = data_root
        self.sf_cfg = self.to_sf_cfg(cfg)

        # The other parameters also need to be "moved" into the config
        optim_str = str(type(optimizer).__name__).split(".")[-1].lower()
        self.sf_cfg.optimizer = optim_str
        # TODO: Potentially extract parameters from brain if not in cfg?! For now just warn it's ignored:
        if brain is not None or train_set is not None or test_set is not None or completed_epochs is not None or histories is not None:
            warnings.warn("brain, train_set, test_set, completed_epochs and histories can not (yet) be set and are ignored")

        # we need to convert to the sample_factory config style since we can not change the function signatures
        # of the library and that uses it _everywhere_

        # Register retinal environments and models.
        register_retinal_env(self.sf_cfg.env, self.data_root, self.sf_cfg.input_satiety)
        global_model_factory().register_actor_critic_factory(SampleFactoryBrain)
        return brain, optimizer, None, None
    
    def analyze(self, _cfg: DictConfig, device: torch.device, brain: Brain, histories: Dict[str, List[float]], train_set: Dataset[Tuple[Tensor | int]], test_set: Dataset[Tuple[Tensor | int]], epoch: int, copy_checkpoint: bool = False):
        
        status = enjoy(self.sf_cfg)
        return status


def brain_from_actor_critic(actor_critic: SampleFactoryBrain) -> Brain:
    return actor_critic.get_brain()  # TODO: Check if needed
