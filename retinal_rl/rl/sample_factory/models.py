from enum import Enum
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.action_parameterization import get_action_distribution
from sample_factory.algo.utils.action_distributions import (
    calc_num_action_parameters,
)
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.model_utils import model_device
from sample_factory.utils.normalize import ObservationNormalizer
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from torch import Tensor

from retinal_rl.models.brain import Brain
from retinal_rl.models.circuits.actor_critic import Actor, Critic
from retinal_rl.rl.sample_factory.sf_interfaces import ActorCriticProtocol
from runner.util import create_brain  # TODO: Remove runner reference!


class CoreMode(Enum):
    IDENTITY = (0,)
    SIMPLE = (1,)
    RNN = (2,)
    MULTI_MODULES = (3,)


class SampleFactoryBrain(ActorCritic, ActorCriticProtocol):
    MISSING_RNN_STATE_VALUE = 999999  # Placeholder for missing RNN states

    def __init__(self, cfg: Config, obs_space: ObsSpace, action_space: ActionSpace):
        # Attention: make_actor_critic passes [cfg, obs_space, action_space], but ActorCritic takes the reversed order of arguments [obs_space, action_space, cfg]
        super().__init__(obs_space, action_space, cfg)

        self.actor, self.critic = self.check_actor_critic(
            DictConfig(cfg.brain), action_space
        )

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

    def set_brain(self, brain: Brain):
        """
        method to set weights / brain.
        Checks for brain compatibility.
        Decide which part of the brain is head/core/tail or creates Identity transforms if needed.
        """
        self.brain = brain

    @staticmethod
    def check_actor_critic(
        config: DictConfig, action_space: ActionSpace
    ) -> tuple[str, str]:
        actor_target = f"{Actor.__module__}.{Actor.__name__}"
        critic_target = f"{Critic.__module__}.{Critic.__name__}"
        # Check if actor and critic circuits are present in cfg based on their class
        actor_circuit = None
        critic_circuit = None
        for circuit_name, circuit in config.circuits.items():
            assert isinstance(
                circuit, DictConfig
            ), "Circuit config must be a DictConfig"
            assert "_target_" in circuit, "Circuit config must specify a _target_"
            if circuit._target_ == actor_target:
                actor_circuit = circuit_name
            if circuit._target_ == critic_target:
                critic_circuit = circuit_name
        assert (
            actor_circuit is not None
        ), "Actor circuit is required in RL configurations"
        assert (
            critic_circuit is not None
        ), "Critic circuit is required in RL configurations"

        num_action_outputs = calc_num_action_parameters(action_space)
        assert (
            config.circuits[actor_circuit].num_actions == int(num_action_outputs)
        ), f"Output shape of actor doesn't match action space, {config.circuits[actor_circuit].num_actions} != {[int(num_action_outputs)]}"
        return actor_circuit, critic_circuit

    def forward(
        self,
        normalized_obs_dict: dict[str, Tensor],
        rnn_states: Tensor,
        values_only: bool = False,
        action_mask: Optional[Tensor] = None,
        value_response_index: int = 0,
        actor_response_index: int = 0,
        rnn_state_index: int = 0,
    ) -> TensorDict:
        responses = self.brain(
            {"vision": normalized_obs_dict["obs"], "rnn_state": rnn_states}
        )
        # TODO: this dict entry is bound to the config -> bad!

        # Create Sample Factory result dict
        result = TensorDict(
            values=responses[self.critic][value_response_index].squeeze()
        )
        # if values_only: TODO: Not needed, right?
        #     return result

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = responses[self.actor][actor_response_index]

        # Create distribution object based on the prediction of the action parameters
        # NOTE: Only Discrete action spaces are supported
        self.last_action_distribution = get_action_distribution(
            self.action_space,
            raw_logits=responses[self.actor][actor_response_index],
            action_mask=action_mask,
        )
        #  TODO: Check: would be nice to get rid of self.last_action_distribution & self.action_distribution()

        self._maybe_sample_actions(True, result)

        # pipe rnn states through result dict for sample factory
        # TODO: Support for multiple RNN nodes?
        rnn_node = []
        if "rnn_state" in self.brain.connectome.nodes:
            rnn_node = [node for node in self.brain.connectome.successors("rnn_state")]
        if len(rnn_node) == 1:
            core_out = responses[rnn_node[0]][rnn_state_index]
            result["new_rnn_states"] = core_out
            result["latent_states"] = core_out
        elif len(rnn_node) == 0:
            # Use a large constant as a placeholder for missing RNN states.
            result["new_rnn_states"] = torch.full_like(
                rnn_states, self.MISSING_RNN_STATE_VALUE
            )
            # Sample Factory always needs "new_rnn_states" in the output - #TODO: Trace down the usage
        else:
            raise ValueError(
                f"RNN state should have exactly one successor if present, but found {len(rnn_node)}."
            )
        return result

    def get_brain(self) -> Brain:
        return self.brain

    # Methods need to be overwritten 'cause the use .encoders
    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32
