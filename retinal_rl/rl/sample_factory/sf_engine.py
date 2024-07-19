
from typing import Tuple
from omegaconf import DictConfig
import torch

from retinal_rl.models.brain import Brain
from retinal_rl.rl.interface import RLEngine
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from retinal_rl.rl.sample_factory.config_defaults import SfDefaults

class SFEngine(RLEngine):
    def train(
        cfg: DictConfig,
        device: torch.device,
        brain: Brain
    ):
        pass
    
    def unpack_cfg(cfg: DictConfig, brain:Brain) -> Tuple[ObsSpace, ActionSpace, Config]:
        obs_space = ObsSpace() # Get obs space from cfg or brain model
        action_space = ActionSpace() # Get action space from cfg or brain model
        sf_cfg = SfDefaults() # Load Defaults
        # TODO: merge cfg and sf_cfg
        return obs_space, action_space, sf_cfg

def actor_critic_from_brain(brain: Brain) -> ActorCritic:
    assert brain.connectome

def brain_from_actor_critic(actor_critic: ActorCritic) -> ActorCritic: # probably move to model interface
    modules = actor_critic.children()
