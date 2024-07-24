from typing import Dict, Optional, Protocol, Tuple
from sample_factory.utils.typing import ActionSpace, ActionDistribution
from torch import Tensor
import torch
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.action_parameterization import ActionsParameterization
from typing import Protocol

class ActorCriticProtocol(Protocol, ActorCritic):
    """
    Protocol extracted from Sample Factory for Actor Critic, based on usage in Learner.
    This suggests that internally, ActorCritic can be built whatever as long as you assign some function part to be "head|core|tail".
    The inputs must match the outputs of the previous component, obviously.
    """
    action_space: ActionSpace
    returns_normalizer: Optional[RunningMeanStdInPlace]

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor: ...

    def forward_core(
        self, head_output: Tensor, rnn_states: Tensor
    ) -> Tuple[Tensor, Tensor]: ...

    def forward_tail(
        self,
        core_output: Tensor,
        values_only: bool = False,
        sample_actions: bool = False,
    ) -> TensorDict: ...

    def forward(
        self, normalized_obs_dict: Dict[str, Tensor], rnn_states: Tensor, values_only: bool = False
    ) -> TensorDict: ...

    def action_distribution(self) -> ActionDistribution: ...
    """self.last_action_distribution has to be updated in forward_tail"""

    def model_to_device(self, device: torch.device) -> None: ...
    """Actor Critic implements this for all module children"""

    def train(self, mode:bool = True) -> ActorCritic: ...
    """should be implemented through nn.Module()"""

    def summaries(self) -> Dict[str, float]: ...
    """default implementation requires self.obs_normalizer: ObservationNormalizer"""

    def get_action_parameterization(self, decoder_output_size: int) -> ActionsParameterization: ...
    """
    default implementation uses self.cfg
    for discrete action spaces, it defaults to a simple fully connected layer mapping from the tail output to the number of actions.
    (cfg is not really used in that case.)
    """

class ModelConfigProtocol(Protocol):
    """
    This is just to document what is in the config for the model
    """
    normalize_returns: bool
    adaptive_stddev: bool
    policy_init_gain: float
    policy_initialization: str # ("orthogonal", "xavier_uniform", or "torch_default")

    # if get_action_parameterization is not overwritten
    initial_stddev: float
    continuous_tanh_scale: float

    # These are likely to be present based on the usage, but aren't directly referenced in the provided code
    seed: Optional[int]
    learning_rate: float
    train_for_env_steps: int
    batch_size: int
    num_epochs: int
    num_batches_per_epoch: int
    max_policy_lag: int
    summaries_use_frameskip: bool