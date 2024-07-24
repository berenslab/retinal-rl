from typing import Dict, Optional
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.core import ModelCore
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.tensor_dict import TensorDict
from torch import Tensor
import torch
from retinal_rl.models.brain import Brain
from retinal_rl.rl.interface import BrainInterface
from retinal_rl.rl.sample_factory.sf_interfaces import ActorCriticProtocol

def default_cfg():
    ...


class SampleFactoryBrain(ActorCritic, ActorCriticProtocol, BrainInterface):
    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(obs_space, action_space, cfg)

        self.set_brain(Brain(**cfg.brain)) # TODO: Find way to instantiate brain outside

        self.action_parameterization = self.get_action_parameterization(
            self.decoder.get_out_size()
        )

    def set_brain(self, brain: Brain):
        """
        method to set weights / brain.
        Checks for brain compatibility.
        Decide which part of the brain is head/core/tail or creates Identity transforms if needed.
        """
        if self.check_brain_compatible(brain):
            self.brain = brain

    def check_brain_compatible(self, brain: Brain):
        raise NotImplementedError()

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward_core(self, head_output, rnn_states):
        raise NotImplementedError()

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool
    ) -> TensorDict:
        raise NotImplementedError()

    def forward(
        self, normalized_obs_dict, rnn_states, values_only: bool = False
    ) -> TensorDict:
        raise NotImplementedError()

    def get_brain(self) -> Brain:
        return self.brain
