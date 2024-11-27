"""Objectives for training models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from sample_factory.algo.learning.rnn_utils import (
    build_core_out_from_seq,
    build_rnn_inputs,
)
from sample_factory.algo.utils.action_distributions import (
    get_action_distribution,
    is_continuous_action_space,
)
from sample_factory.algo.utils.torch_utils import masked_select
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log
from torch import Tensor

from retinal_rl.models.loss import BaseContext, Loss


class RLContext(BaseContext):
    """Context class for classification tasks.

    This class extends BaseContext with attributes specific to classification problems.

    Attributes
    ----------
        inputs (Tensor): The input data for the classification task.
        classes (Tensor): The true class labels for the input data.

    """

    def __init__(
        self,
        sources: Tensor,
        inputs: Tensor,
        responses: Dict[str, Tensor],
        epoch: int,
        num_invalids: int,
        policy_id: int,
        # attributes for loss calculation
        # TODO: types
        ratio: float,
        adv,
        valids,
        action_distribution,
        action_distribution_parameter,
        values,
        old_values,
        targets,
    ):
        super().__init__(sources, inputs, responses, epoch)
        self.num_invalids = num_invalids
        self.policy_id = policy_id  # TODO: Why needed?

        # TODO: additional attributes
        self.ratio = ratio
        self.adv = adv
        self.valids = valids
        self.action_distribution = action_distribution
        self.action_distribution_parameter = action_distribution_parameter
        self.values = values
        self.old_values = old_values
        self.targets = targets


class PolicyLoss(Loss[RLContext]):
    """TODO: Doc"""

    def __init__(
        self,
        target_circuits: List[str] = [],
        weights: List[float] = [],
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        clip_ratio=1.1,  # TODO: Check default
    ):
        # PPO clipping
        self.clip_ratio_high = 1.0 + clip_ratio  # e.g. 1.1
        # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
        self.clip_ratio_low = 1.0 / self.clip_ratio_high

    @staticmethod
    def _policy_loss(
        ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids: int
    ):
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
        loss_unclipped = ratio * adv
        loss_clipped = clipped_ratio * adv
        loss = torch.min(loss_unclipped, loss_clipped)
        loss = masked_select(loss, valids, num_invalids)
        return -loss.mean()

    def compute_value(self, context):
        return self._policy_loss(
            context.ratio,
            context.adv,
            self.clip_ratio_low,
            self.clip_ratio_high,
            context.valids,
            context.num_invalids,
        )


class ExplorationLoss(Loss[RLContext]):
    """TODO: Doc"""

    def __init__(
        self,
        target_circuits: List[str] = [],
        weights: List[float] = [],
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        loss_coeff: float = 1.0,  # TODO: This is taken care of by the objective system
        exploration_loss: str = "entropy",  # TODO: Different losses for this?!
    ):
        self.loss_coeff = loss_coeff
        if self.exploration_loss_coeff == 0.0:
            self.exploration_loss_func = lambda action_distr, valids, num_invalids: 0.0
        elif exploration_loss == "entropy":
            self.exploration_loss_func = self._entropy_exploration_loss
        elif exploration_loss == "symmetric_kl":
            self.exploration_loss_func = self._symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f"{exploration_loss} not supported!")

    def _entropy_exploration_loss(
        self, action_distribution, valids, num_invalids: int
    ) -> Tensor:
        entropy = action_distribution.entropy()
        entropy = masked_select(entropy, valids, num_invalids)
        return -self.loss_coeff * entropy.mean()

    def _symmetric_kl_exploration_loss(
        self, action_distribution, valids, num_invalids: int
    ) -> Tensor:
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = masked_select(kl_prior, valids, num_invalids).mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        return self.loss_coeff * kl_prior

    def compute_value(self, context):
        return self.exploration_loss_func(
            context.action_distribution, context.valids, context.num_invalids
        )


class KLLoss(Loss[RLContext]):
    """TODO: Doc"""

    def __init__(
        self,
        target_circuits: List[str] = [],
        weights: List[float] = [],
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        action_space=None,  # TODO: Where to get this
        loss_coeff: float = 0.0,  # TODO: This is taken care of by the objective system
    ):
        self.action_space = action_space
        self.loss_coeff = loss_coeff  # TODO: loss coefficient system
        if loss_coeff == 0.0:
            if is_continuous_action_space(self.env_info.action_space):
                log.warning(
                    "WARNING! It is generally recommended to enable Fixed KL loss (https://arxiv.org/pdf/1707.06347.pdf) for continuous action tasks to avoid potential numerical issues. "
                    "I.e. set --kl_loss_coeff=0.1"
                )
            self.kl_loss_func = (
                lambda action_space,
                action_logits,
                distribution,
                valids,
                num_invalids: 0.0
            )
        else:
            self.kl_loss_func = self._kl_loss

    def _kl_loss(
        self,
        action_space,
        action_logits,
        action_distribution,
        valids,
        num_invalids: int,
    ) -> Tuple[Tensor, Tensor]:
        old_action_distribution = get_action_distribution(action_space, action_logits)
        kl_old = action_distribution.kl_divergence(old_action_distribution)
        kl_old = masked_select(kl_old, valids, num_invalids)
        kl_loss = kl_old.mean()

        kl_loss *= self.loss_coeff

        return kl_old, kl_loss

    def compute_value(self, context):
        # TODO: this returns kl_loss, kl_loss_old
        return self.kl_loss_func(
            self.action_space,
            context.action_distribution_parameter,
            context.action_distribution,
            context.valids,
            context.num_invalids,
        )


class ValueLoss(Loss[RLContext]):
    """TODO: Doc"""

    def __init__(
        self,
        target_circuits: List[str] = [],
        weights: List[float] = [],
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        clip_value=1.0,  # TODO: Check clip value default
        loss_coeff: float = 1.0,  # TODO: This is taken care of by the objective system
    ):
        self.clip_value = clip_value
        self.loss_coeff = loss_coeff

    def _value_loss(
        self,
        new_values: Tensor,
        old_values: Tensor,
        target: Tensor,
        clip_value: float,
        valids: Tensor,
        num_invalids: int,
    ) -> Tensor:
        value_clipped = old_values + torch.clamp(
            new_values - old_values, -clip_value, clip_value
        )
        value_original_loss = (new_values - target).pow(2)
        value_clipped_loss = (value_clipped - target).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss)
        value_loss = masked_select(value_loss, valids, num_invalids)
        value_loss = value_loss.mean()

        value_loss *= self.loss_coeff

        return value_loss

    def compute_value(self, context):
        # TODO: Unpack context
        return self._value_loss(
            context.values,
            context.old_values,
            context.targets,
            self.clip_value,
            context.valids,
            context.num_invalids,
        )


@dataclass
class VTraceParams:
    with_vtrace: bool
    vtrace_rho: float
    vtrace_c: float
    gamma: float


def build_context(
    actor_critic: ActorCritic,
    mb: AttrDict,
    num_invalids: int,
    epoch: int,
    policy_id: int,
    recurrence: int,
    timing: Timing,
    use_rnn: bool,
    vtrace_params: VTraceParams,
    device: torch.device,
) -> RLContext:
    with torch.no_grad(), timing.add_time("losses_init"):
        recurrence: int = recurrence
        valids = mb.valids

    # calculate policy head outside of recurrent loop
    with timing.add_time("forward_head"):
        head_outputs = actor_critic.forward_head(mb.normalized_obs)
        minibatch_size: int = head_outputs.size(0)

    # initial rnn states
    with timing.add_time("bptt_initial"):
        if use_rnn:
            # this is the only way to stop RNNs from backpropagating through invalid timesteps
            # (i.e. experience collected by another policy)
            done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()
            head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                head_outputs,
                done_or_invalid,
                mb.rnn_states,
                recurrence,
            )
        else:
            rnn_states = mb.rnn_states[::recurrence]

    # calculate RNN outputs for each timestep in a loop
    with timing.add_time("bptt"):
        if use_rnn:
            with timing.add_time("bptt_forward_core"):
                core_output_seq, _ = actor_critic.forward_core(
                    head_output_seq, rnn_states
                )
            core_outputs = build_core_out_from_seq(
                core_output_seq, inverted_select_inds
            )
            del core_output_seq
        else:
            core_outputs, _ = actor_critic.forward_core(head_outputs, rnn_states)

        del head_outputs

    num_trajectories = minibatch_size // recurrence
    assert core_outputs.shape[0] == minibatch_size

    with timing.add_time("tail"):
        # calculate policy tail outside of recurrent loop
        result = actor_critic.forward_tail(
            core_outputs, values_only=False, sample_actions=False
        )
        action_distribution = actor_critic.action_distribution()
        log_prob_actions = action_distribution.log_prob(mb.actions)
        ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

        # super large/small values can cause numerical problems and are probably noise anyway
        ratio = torch.clamp(ratio, 0.05, 20.0)

        values = result["values"].squeeze()

        del core_outputs

    # these computations are not the part of the computation graph
    with torch.no_grad(), timing.add_time("advantages_returns"):
        if vtrace_params.with_vtrace:
            # V-trace parameters
            rho_hat = torch.Tensor([vtrace_params.vtrace_rho])
            c_hat = torch.Tensor([vtrace_params.vtrace_c])

            ratios_cpu = ratio.cpu()
            values_cpu = values.cpu()
            rewards_cpu = mb.rewards_cpu
            dones_cpu = mb.dones_cpu

            vtrace_rho = torch.min(rho_hat, ratios_cpu)
            vtrace_c = torch.min(c_hat, ratios_cpu)

            vs = torch.zeros(num_trajectories * recurrence)
            adv = torch.zeros(num_trajectories * recurrence)

            next_values = (
                values_cpu[recurrence - 1 :: recurrence]
                - rewards_cpu[recurrence - 1 :: recurrence]
            )
            next_values /= vtrace_params.gamma
            next_vs = next_values

            for i in reversed(range(recurrence)):
                rewards = rewards_cpu[i::recurrence]
                dones = dones_cpu[i::recurrence]
                not_done = 1.0 - dones
                not_done_gamma = not_done * vtrace_params.gamma

                curr_values = values_cpu[i::recurrence]
                curr_vtrace_rho = vtrace_rho[i::recurrence]
                curr_vtrace_c = vtrace_c[i::recurrence]

                delta_s = curr_vtrace_rho * (
                    rewards + not_done_gamma * next_values - curr_values
                )
                adv[i::recurrence] = curr_vtrace_rho * (
                    rewards + not_done_gamma * next_vs - curr_values
                )
                next_vs = (
                    curr_values
                    + delta_s
                    + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                )
                vs[i::recurrence] = next_vs

                next_values = curr_values

            targets = vs.to(device)
            adv = adv.to(device)
        else:
            # using regular GAE
            adv = mb.advantages
            targets = mb.returns

        adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
        adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage

    return RLContext(
        None,
        None,
        None,
        epoch,
        num_invalids,
        policy_id,
        ratio,
        adv,
        valids,
        action_distribution,
        mb.action_logits,
        values,
        mb.values,
        targets,
    )
