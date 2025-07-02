"""Objectives for training models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
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
        super().__init__(target_circuits, weights, min_epoch, max_epoch)
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
        exploration_loss: str = "symmetric_kl",  # TODO: Different losses for this?!
    ):
        super().__init__(target_circuits, weights, min_epoch, max_epoch)
        if exploration_loss == "entropy":
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
        return -entropy.mean()

    def _symmetric_kl_exploration_loss(
        self, action_distribution, valids, num_invalids: int
    ) -> Tensor:
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = masked_select(kl_prior, valids, num_invalids).mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        return torch.clamp(kl_prior, max=30)

    def compute_value(self, context):
        return self.exploration_loss_func(
            context.action_distribution, context.valids, context.num_invalids
        )


class KlLoss(Loss[RLContext]):
    """TODO: Doc"""

    def __init__(
        self,
        target_circuits: List[str] = [],
        weights: List[float] = [],
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        action_space=None,  # TODO: Where to get this
    ):
        super().__init__(target_circuits, weights, min_epoch, max_epoch)
        self.action_space = action_space
        if all(x == 0.0 for x in weights) and is_continuous_action_space(
            self.action_space
        ):
            log.warning(
                "WARNING! It is generally recommended to enable Fixed KL loss (https://arxiv.org/pdf/1707.06347.pdf) for continuous action tasks to avoid potential numerical issues. "
                "I.e. set --kl_loss_coeff=0.1"
            )

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

        return kl_old, kl_loss

    def compute_value(self, context):
        # TODO: this theoretically returns kl_loss, kl_loss_old
        return self._kl_loss(
            self.action_space,
            context.action_distribution_parameter,
            context.action_distribution,
            context.valids,
            context.num_invalids,
        )[1]


class ValueLoss(Loss[RLContext]):
    """TODO: Doc"""

    def __init__(
        self,
        target_circuits: List[str] = [],
        weights: List[float] = [],
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        clip_value=0.2,  # TODO: Check clip value default
    ):
        super().__init__(target_circuits, weights, min_epoch, max_epoch)
        self.clip_value = clip_value

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
        return value_loss.mean()

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
    # FIXME: IMPROVISED FOR ENABLING THE MULTI-OBJECTIVE
    # build brain input
    brain_inp = {"vision": mb.normalized_obs["obs"]}

    # TODO: use_rnn not really needed anymore -> check if sample-factory uses it
    if use_rnn:
        rnn_states = mb.rnn_states

        # Add some attributes to rnn_states needed when training to correctly batch & cut gradients
        # low importance TODO: Avoid attaching attributes to an object and us it as an information pipe
        brain_inp.update(
            {
                "rnn_state": {  # TODO: this dict entry is bound to the config -> bad!
                    "states": rnn_states,
                    "valids": mb.valids,
                    "dones_cpu": mb.dones_cpu,
                    "recurrence": recurrence,
                }  # TODO: Confusing as this will get returned in the responses dict as well, at least rename to rnn_inp_state
            }
        )

    # actual forward pass
    responses = actor_critic.brain(brain_inp)

    minibatch_size: int = responses["vision"][0].size(0) #TODO: Also here just accessing 0 is a bit hacky
    num_trajectories = minibatch_size // recurrence

    with timing.add_time("tail"):
        values = responses["critic"][0].squeeze()

        # Get Action Distribution from actor output
        action_distribution = get_action_distribution(
            actor_critic.action_space, raw_logits=responses["actor"][0]
        )
        log_prob_actions = action_distribution.log_prob(mb.actions)
        ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

        # super large/small values can cause numerical problems and are probably noise anyway
        ratio = torch.clamp(ratio, 0.05, 20.0)

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
                # TODO: Check behaviour here when no RNN but recurrence != 0

                next_values = curr_values

            targets = vs.to(device)
            adv = adv.to(device)
        else:
            # using regular GAE
            adv = mb.advantages
            targets = mb.returns

        adv_std, adv_mean = torch.std_mean(masked_select(adv, mb.valids, num_invalids))
        adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage

        vis_input = (
            mb.normalized_obs["obs_raw"]
            if "obs_raw" in mb.normalized_obs
            else mb.normalized_obs["obs"]
        )
    return RLContext(
        vis_input,
        None,
        responses,
        epoch,
        num_invalids,
        policy_id,
        ratio,
        adv,
        mb.valids,
        action_distribution,
        mb.action_logits,
        values,
        mb.values,
        targets,
    )
