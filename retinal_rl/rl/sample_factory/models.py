from enum import Enum
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.action_parameterization import get_action_distribution
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.model_utils import model_device
from sample_factory.utils.normalize import ObservationNormalizer
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

        self.check_brain_config(DictConfig(cfg.brain))  # TODO: Use this

        self.set_brain(create_brain(DictConfig(cfg.brain)))
        # TODO: Find way to instantiate brain outside
        transforms_list = hydra.utils.instantiate(cfg.transforms)
        self.inp_transforms = torch.nn.Sequential(*transforms_list)

    def normalize_obs(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        This is used to implement input transforms!
        """
        if self.cfg[
            "normalize_input"
        ]:  # FIXME: have a properly defined switch between default samplefactory inp normalization and our input transforms
            return self.obs_normalizer(obs)

        obs_clone = ObservationNormalizer._clone_tensordict(obs)
        for k in (
            obs
        ):  # There should be only one key "obs" in all our cases as far as I know
            inp = obs[k].clone() if obs[k].dtype == torch.float else obs[k].float()
            # TODO: This should not be needed after the cloning before
            obs_clone[k] = self.inp_transforms(inp)
            obs_clone[k + "_raw"] = obs[k]
        return obs_clone

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
        result = TensorDict(values=responses["critic"][0].squeeze()) # TODO: Just accessing 0 is a bit hacky
        # if values_only: TODO: Not needed, right?
        #     return result

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = responses["actor"][0]

        # Create distribution object based on the prediction of the action parameters
        # NOTE: Only Discrete action spaces are supported
        self.last_action_distribution = get_action_distribution(
            self.action_space, raw_logits=responses["actor"][0], action_mask=action_mask
        )
        #  TODO: Check: would be nice to get rid of self.last_action_distribution & self.action_distribution()

        self._maybe_sample_actions(True, result)

        # TODO: hack piping the rnn_state through the result dict
        if "rnn_state" in responses:
            core_out = responses["rnn_state"][0]
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
