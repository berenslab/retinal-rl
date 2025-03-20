from enum import Enum
from typing import Optional

import torch
from omegaconf import DictConfig
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.action_parameterization import get_action_distribution
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.model_utils import model_device
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from torch import Tensor

from retinal_rl.models.brain import Brain
from retinal_rl.models.circuits.latent_core import LatentRNN
from retinal_rl.rl.sample_factory.rnn_decorator import decorate_forward
from retinal_rl.rl.sample_factory.sf_interfaces import ActorCriticProtocol
from runner.util import create_brain  # TODO: Remove runner reference!


class CoreMode(Enum):
    IDENTITY = (0,)
    SIMPLE = (1,)
    RNN = (2,)
    MULTI_MODULES = (3,)


class SampleFactoryBrain(ActorCritic, ActorCriticProtocol):
    def __init__(self, cfg: Config, obs_space: ObsSpace, action_space: ActionSpace):
        # Attention: make_actor_critic passes [cfg, obs_space, action_space], but ActorCritic takes the reversed order of arguments [obs_space, action_space, cfg]
        super().__init__(obs_space, action_space, cfg)

        self.check_brain_config(DictConfig(cfg.brain)) # TODO: Use this

        self.set_brain(create_brain(DictConfig(cfg.brain)))
        # TODO: Find way to instantiate brain outside

    def wrap_rnns(self):
        for circuit_name in self.brain.circuits:
            if isinstance(self.brain.circuits[circuit_name], LatentRNN):
                decorate_forward(self.brain.circuits[circuit_name])

    def set_brain(self, brain: Brain):
        """
        method to set weights / brain.
        Checks for brain compatibility.
        Decide which part of the brain is head/core/tail or creates Identity transforms if needed.
        """
        self.brain = brain
        self.wrap_rnns()

    @staticmethod
    def check_brain_config(config: DictConfig):
        assert (
            "critic" in config["circuits"]
        ), "For RL, a circuit named 'critic' is needed"
        assert (
            "actor" in config["circuits"]
        ), "For RL, a circuit named 'actor' is needed"

    def forward(
        self,
        normalized_obs_dict: dict[str, Tensor],
        rnn_states: Tensor,
        values_only: bool = False,
        action_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        responses = self.brain(
            {"vision": normalized_obs_dict["obs"], "rnn_state": rnn_states}
        )
        # TODO: this dict entry is bound to the config -> bad!

        # Create Sample Factory result dict
        result = TensorDict(values=responses["critic"].squeeze())
        # if values_only: TODO: Not needed, right?
        #     return result

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = responses["actor"]

        # Create distribution object based on the prediction of the action parameters
        # NOTE: Only Discrete action spaces are supported
        self.last_action_distribution = get_action_distribution(
            self.action_space, raw_logits=responses["actor"], action_mask=action_mask
        )
        #  TODO: Check: would be nice to get rid of self.last_action_distribution & self.action_distribution()

        self._maybe_sample_actions(True, result)

        # TODO: hack piping the rnn_state through the result dict
        if "rnn_state" in responses:
            core_out = responses["rnn_state"]
            result["new_rnn_states"] = core_out
            result["latent_states"] = core_out
        else:
            result["new_rnn_states"] = torch.full_like(rnn_states, 999999)
            # Sample Factory always needs "new_rnn_states" in the output - #TODO: Trace down the usage
        return result

    def get_brain(self) -> Brain:
        return self.brain

    # Methods need to be overwritten 'cause the use .encoders
    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32
