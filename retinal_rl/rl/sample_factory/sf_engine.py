
from omegaconf import DictConfig
import torch

from retinal_rl.models.brain import Brain
from retinal_rl.rl.interface import RLEngine
from sample_factory.model.actor_critic import ActorCritic

class SFEngine(RLEngine):
    def train(
        cfg: DictConfig,
        device: torch.device,
        brain: Brain
    ):
        pass

def actor_critic_from_brain(brain: Brain) -> ActorCritic:
    assert brain.connectome