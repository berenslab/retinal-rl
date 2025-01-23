import warnings
from enum import Enum
from typing import Optional

import networkx as nx
import numpy as np
import torch
from omegaconf import DictConfig
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.model_utils import model_device
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from torch import Tensor, nn

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

        # self.check_brain_config(DictConfig(cfg.brain)) # TODO: Use this

        self.set_brain(create_brain(DictConfig(cfg.brain)))
        # TODO: Find way to instantiate brain outside

        dec_out_shape = self.brain.circuits[self.decoder].output_shape
        decoder_out_size = int(np.prod(dec_out_shape))
        self.critic_linear = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(
            decoder_out_size
        )
        # boils down to a linear layer mapping to num_action_outputs

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
        self.decoder = self.get_action_decoder(brain)
        self.wrap_rnns()

    @staticmethod
    def check_brain_config(config: DictConfig):
        assert (
            "critic" in config["circuits"]
        ), "For RL, a circuit named 'critic' is needed"
        assert (
            "actor" in config["circuits"]
        ), "For RL, a circuit named 'actor' is needed"
        assert (
            "latent" in config["circuits"]
        ), "For RL, a circuit named 'latent' is needed"

    @staticmethod
    def get_action_decoder(brain: Brain) -> str:
        assert "vision" in brain.sensors  # needed as input
        # potential TODO: add other input sources if needed?

        vision_paths: list[list[str]] = []
        for node in brain.connectome:
            if brain.connectome.out_degree(node) == 0:  # it's a leaf
                vision_paths.append(nx.shortest_path(brain.connectome, "vision", node))

        decoder = "action_decoder"  # default assumption
        if decoder in brain.circuits:  # needed to produce output = decoder
            vision_path: list[str] = nx.shortest_path(
                brain.connectome, "vision", "action_decoder"
            )
        else:
            selected_path = 0
            out_dim = None
            for i, vision_path in enumerate(
                vision_paths
            ):  # Assuming that the path with the smallest output dimension is best for action prediction (eg in contrast to a decoder trying to reproduce the input)
                dec_out_shape = brain.circuits[vision_path[-1]].output_shape
                if not (out_dim) or np.prod(dec_out_shape) < out_dim:
                    out_dim = int(np.prod(dec_out_shape))
                    selected_path = i
            vision_path = vision_paths[selected_path]
            warnings.warn(
                message="No action_decoder in model. Will use "
                + vision_path[-1]
                + " instead."
            )
            decoder = vision_path[-1]

        return decoder

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

        out = torch.flatten(responses[self.decoder], 1)

        # TODO: make critic_linear RL-Style
        values = self.critic_linear(out).squeeze()

        result = TensorDict(values=values)
        # if values_only: FIXME:
        #     return result

        # TODO: make action parameterization RL-Style
        action_distribution_params, self.last_action_distribution = (
            self.action_parameterization(out, action_mask)
        )

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(True, result)

        # TODO: Check why both is needed but the same in all cases I looked at so far
        if "rnn" in responses:
            core_out = responses["rnn"]
            result["new_rnn_states"] = core_out
            result["latent_states"] = core_out
        else:
            result["new_rnn_states"] = torch.full_like(
                rnn_states, 999999
            )  # Sample Factory always needs "new_rnn_states" in the output - TODO: Trace down the usage
        return result

    def get_brain(self) -> Brain:
        return self.brain

    # Methods need to be overwritten 'cause the use .encoders
    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32
