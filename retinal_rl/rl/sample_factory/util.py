import json
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

# from retinal_rl.rl.sample_factory.observer import RetinalAlgoObserver
import torch
from omegaconf import DictConfig
from sample_factory.algo.learning.learner import Learner
from sample_factory.cfg.arguments import (
    load_from_checkpoint,
)
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config

from retinal_rl.models.brain import Brain
from runner.util import create_brain


def get_checkpoint(cfg: Config) -> tuple[Dict[str, Any], AttrDict]:
    # verbose = False

    cfg = load_from_checkpoint(cfg)

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    policy_id = cfg.policy_index
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, policy_id), "checkpoint_*"
    )
    best_checkpoint = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, policy_id), "best_*"
    ) # If only a best chekpoint is availabe, use it
    checkpoints.extend(best_checkpoint)
    checkpoint_dict: Dict[str, Any] = Learner.load_checkpoint(checkpoints, device)

    return checkpoint_dict, cfg

@staticmethod
def load_brain_from_checkpoint(
    config: Union[str, Config],
    load_weights: bool = True,
    device: Optional[torch.device] = None,
) -> Brain:
    if isinstance(config, str):
        with open(os.path.join(config, "config.json")) as f:
            config = Namespace(**json.load(f))
    checkpoint_dict, config = get_checkpoint(config)
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
