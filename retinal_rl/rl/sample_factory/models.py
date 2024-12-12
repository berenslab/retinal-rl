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

        self.set_brain(create_brain(DictConfig(cfg.brain)))
        # TODO: Find way to instantiate brain outside

        dec_out_shape = self.brain.circuits[self.decoder_names[-1]].output_shape
        decoder_out_size = int(np.prod(dec_out_shape))
        self.critic_linear = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(
            decoder_out_size
        )
        # boils down to a linear layer mapping to num_action_outputs

    def set_brain(self, brain: Brain):
        """
        method to set weights / brain.
        Checks for brain compatibility.
        Decide which part of the brain is head/core/tail or creates Identity transforms if needed.
        """
        enc, core, dec = self.get_encoder_decoder(brain)
        self.brain = brain
        self.encoder_names = enc
        self.core_mode = core
        self.decoder_names = dec

    @staticmethod
    def get_encoder_decoder(brain: Brain) -> tuple[list[str], CoreMode, list[str]]:
        assert "vision" in brain.sensors  # needed as input
        # potential TODO: add other input sources if needed?

        vision_paths: list[list[str]] = []
        for node in brain.connectome:
            if brain.connectome.out_degree(node) == 0:  # it's a leaf
                vision_paths.append(nx.shortest_path(brain.connectome, "vision", node))

        decoder = "action_decoder"  # default assumption
        if decoder in brain.circuits:  # needed to produce output = decoder
            vision_path: list[str] = nx.shortest_path(brain.connectome, "vision", "action_decoder")
        else:
            selected_path = 0
            out_dim = None
            for i, vision_path in enumerate(
                vision_paths
            ):  # Assuming that the path with the smallest output dimension is best for action prediction (eg in contrast to a decoder trying to reproduce the input)
                dec_out_shape = brain.circuits[vision_path[-1]].output_shape
                if not(out_dim) or np.prod(dec_out_shape) < out_dim:
                    out_dim = int(np.prod(dec_out_shape))
                    selected_path = i
            vision_path = vision_paths[selected_path]
            warnings.warn(
                message = "No action_decoder in model. Will use " + vision_path[-1] + " instead."
            )

        core = "core" # default assumption
        if core in vision_path:
            encoder_end = vision_path.index(core)

            core_mode = CoreMode.SIMPLE
            CoreMode.SIMPLE.value = core
            encoder = vision_path[1:encoder_end]
            decoder = vision_path[encoder_end:-1]
        else: # TODO: Multi Module and RNN Core
            core_mode= CoreMode.IDENTITY
            encoder = vision_path[1:-1]
            decoder = [vision_path[-1]]
            warnings.warn("Seems like there is no model core. Will use an Identity.")

        return encoder, core_mode, decoder

    def forward_head(self, normalized_obs_dict: dict[str, Tensor]) -> Tensor:
        out = normalized_obs_dict["obs"]
        for enc in self.encoder_names:
            out = self.brain.circuits[enc](out)
        return out

    def forward_core(self, head_output: Tensor, rnn_states: Tensor):
        # TODO: what to do with rnn states? -> implement neural circuit that is capable?! (could also be done by some module internally, right?)
        if self.core_mode == CoreMode.SIMPLE:
            out = self.brain.circuits[str(self.core_mode.value)](head_output)
        elif self.core_mode == CoreMode.RNN:
            out, rnn_states = self.brain.circuits[str(self.core_mode.value)](
                head_output, rnn_states
            )
        elif self.core_mode == CoreMode.IDENTITY:
            out = head_output
        else: #self.core_mode == CoreMode.MULTI_MODULES:
            out = head_output
            for m in self.core_mode.value:
                out = self.brain.circuits[str(m)](out)
        return out, rnn_states

    def forward_tail(
        self,
        core_output: Tensor,
        values_only: bool,
        sample_actions: bool,
        action_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        out = core_output
        for m in self.decoder_names:
            out = self.brain.circuits[m](core_output)
        out = torch.flatten(out, 1)

        values = self.critic_linear(out).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = (
            self.action_parameterization(out)
        )

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self,
        normalized_obs_dict: dict[str, Tensor],
        rnn_states: Tensor,
        values_only: bool = False,
        action_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        head_out = self.forward_head(normalized_obs_dict)
        core_out, new_rnn_states = self.forward_core(head_out, rnn_states)
        result = self.forward_tail(
            core_out, values_only=values_only, sample_actions=True
        )
        result["new_rnn_states"] = new_rnn_states
        result["latent_states"] = core_out
        return result

    def get_brain(self) -> Brain:
        return self.brain

    # Methods need to be overwritten 'cause the use .encoders
    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32
