
from typing import Tuple
from omegaconf import DictConfig
import torch

from retinal_rl.models.brain import Brain
from retinal_rl.rl.interface import RLEngine
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from retinal_rl.rl.sample_factory.config_defaults import SfDefaults
from sample_factory.algo.utils.context import global_model_factory
from retinal_rl.rl.system.environment import register_retinal_env

from retinal_rl.rl.system.exec import run_rl

class SFEngine(RLEngine):
    def train(
            self,
            cfg: DictConfig,
            device: torch.device,
    ):
        sf_cfg = self.unpack_cfg(cfg) # we need to convert to the sample_factory config style since we can not change the function signatures of the library and that uses it _everywhere_

        # Register retinal environments and models.
        register_retinal_env(sf_cfg.env, sf_cfg.input_satiety)
        global_model_factory().register_actor_critic_factory(SampleFactoryBrain)

        # Run simulation
        if not (cfg.dry_run):

            status = run_rl(cfg)
            return status
        pass
    
    def unpack_cfg(cfg: DictConfig) -> Tuple[ObsSpace, ActionSpace, Config]:
        obs_space = ObsSpace() # Get obs space from cfg or brain model
        action_space = ActionSpace() # Get action space from cfg or brain model
        sf_cfg = SfDefaults() # Load Defaults
        # TODO: merge cfg and sf_cfg
        sf_cfg.brain = cfg.brain
        return obs_space, action_space, sf_cfg

def actor_critic_from_brain(brain: Brain) -> ActorCritic:
    assert brain.connectome

def brain_from_actor_critic(actor_critic: ActorCritic) -> ActorCritic: # probably move to model interface
    modules = actor_critic.children()
