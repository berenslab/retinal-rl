from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from sample_factory.algo.learning.learner import (
    Learner,
    LearningRateScheduler,
    get_lr_scheduler,
    model_initialization_data,
)
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import (
    LEARNER_ENV_STEPS,
    POLICY_ID_KEY,
    STATS_KEY,
    TRAIN_STATS,
    memory_stats,
)
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.optimizers import Lamb
from sample_factory.algo.utils.rl_utils import gae_advantages, prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.cfg.configurable import Configurable
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.decay import LinearDecay
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import (
    Config,
    InitModelData,
    PolicyID,
)
from sample_factory.utils.utils import log
from torch import Tensor

from retinal_rl.models.objective import Objective
from retinal_rl.rl.loss import KlLoss, RLContext, VTraceParams, build_context
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain

logger = logging.getLogger(__name__)


class RetinalLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        Configurable.__init__(self, cfg)

        self.timing = Timing(name=f"Learner {policy_id} profile")

        self.policy_id = policy_id

        self.env_info = env_info

        self.device = None
        self.actor_critic: Optional[ActorCritic] = None

        self.optimizer = None

        self.objective: Objective[RLContext] = None

        self.curr_lr: Optional[float] = None
        self.lr_scheduler: Optional[LearningRateScheduler] = None

        self.train_step: int = 0  # total number of SGD steps
        self.env_steps: int = 0  # total number of environment steps consumed by the learner

        self.best_performance = -1e9

        # for configuration updates, i.e. from PBT
        self.new_cfg: Optional[Dict] = None

        # for multi-policy learning (i.e. with PBT) when we need to load weights of another policy
        self.policy_to_load: Optional[PolicyID] = None

        # decay rate at which summaries are collected
        # save summaries every 5 seconds in the beginning, but decay to every 4 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self.summary_rate_decay_seconds = LinearDecay([(0, 2), (100000, 60), (1000000, 120)])
        self.last_summary_time = 0
        self.last_milestone_time = 0

        # shared tensor used to share the latest policy version between processes
        self.policy_versions_tensor: Tensor = policy_versions_tensor

        self.param_server: ParameterServer = param_server

        self.exploration_loss_func: Optional[Callable] = None
        self.kl_loss_func: Optional[Callable] = None

        self.is_initialized = False

    def init(self) -> InitModelData:
        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # initialize device
        self.device = policy_device(self.cfg, self.policy_id)

        log.debug("Initializing actor-critic model on device %s", self.device)

        # trainable torch module
        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        # TODO: Check actor_critic usage
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self.actor_critic)
        self.actor_critic.model_to_device(self.device)

        def share_mem(t):
            if t is not None and not t.is_cuda:
                return t.share_memory_()
            return t

        # noinspection PyProtectedMember
        self.actor_critic._apply(share_mem)
        self.actor_critic.train()

        params = list(self.actor_critic.parameters())

        optimizer_cls = dict(adam=torch.optim.Adam, lamb=Lamb)  # TODO: Support for other optimizers
        if self.cfg.optimizer not in optimizer_cls:
            raise RuntimeError(f"Unknown optimizer {self.cfg.optimizer}")

        optimizer_cls = optimizer_cls[self.cfg.optimizer]
        log.debug(f"Using optimizer {optimizer_cls}")

        optimizer_kwargs = dict(
            lr=self.cfg.learning_rate,  # use default lr only in ctor, then we use the one loaded from the checkpoint
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        if self.cfg.optimizer in ["adam", "lamb"]:
            optimizer_kwargs["eps"] = self.cfg.adam_eps

        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

        assert isinstance(self.actor_critic, SampleFactoryBrain) # for now let's just assert that
        self.objective = instantiate(self.cfg.objective, brain=self.actor_critic.brain)

        # Hotfix: inject action space to kl_loss TODO: Fix this
        for loss in self.objective.losses:
            if isinstance(loss, KlLoss):
                loss.action_space = self.env_info.action_space

        self.load_from_checkpoint(self.policy_id)
        self.param_server.init(self.actor_critic, self.train_step, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        self.lr_scheduler = get_lr_scheduler(self.cfg)
        self.curr_lr = self.cfg.learning_rate if self.curr_lr is None else self.curr_lr
        self._apply_lr(self.curr_lr)

        self.is_initialized = True

        return model_initialization_data(self.cfg, self.policy_id, self.actor_critic, self.train_step, self.device)

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict["train_step"]
            self.env_steps = checkpoint_dict["env_steps"]
            self.best_performance = checkpoint_dict.get("best_performance", self.best_performance)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.curr_lr = checkpoint_dict.get("curr_lr", self.cfg.learning_rate)

        log.info(f"Loaded experiment state at {self.train_step=}, {self.env_steps=}")

    def _get_checkpoint_dict(self):
        checkpoint = {
            "train_step": self.train_step,
            "env_steps": self.env_steps,
            "best_performance": self.best_performance,
            "model": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "curr_lr": self.curr_lr,
        }
        return checkpoint

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

    def _train(
        self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num, indices in enumerate(minibatches):
                with torch.no_grad(), timing.add_time("minibatch_init"):

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                # Forward pass through the model / calculate losses
                vtrace_params = VTraceParams(self.cfg.with_vtrace, self.cfg.vtrace_rho, self.cfg.vtrace_c, self.cfg.gamma)
                context = build_context(self.actor_critic, mb, num_invalids, epoch, self.policy_id, self.cfg.recurrence, timing, self.cfg.use_rnn, vtrace_params, self.device) # TODO: transform mb to inputs & sources!

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # TODO: KL Old not returned in our loss, so recalculate it here? Additional Log Statistic?
                    # if kl_old is not None it is already calculated above
                    # if kl_old is None:
                    # calculate KL-divergence with the behaviour policy action distribution
                    old_action_distribution = get_action_distribution(
                        self.actor_critic.action_space,
                        mb.action_logits,
                    )
                    kl_old = context.action_distribution.kl_divergence(old_action_distribution)
                    kl_old = masked_select(kl_old, mb.valids, num_invalids)

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                # update the weights
                with timing.add_time("update"):
                    self.optimizer.zero_grad(set_to_none=True)

                    loss_dict = self.objective.backward(context)
                    # action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries = loss_dict # TODO: etc

                    # TODO: log here based on loss_dict instead of inside losses function
                    with timing.add_time("losses_postprocess"):
                        # noinspection PyTypeChecker
                        actor_loss: float = loss_dict['policy_loss'] + loss_dict['exploration_loss'] + loss_dict['kl_loss']
                        critic_loss = loss_dict['value_loss']
                        loss: float = actor_loss + critic_loss

                        epoch_actor_losses[batch_num] = float(actor_loss)

                        high_loss = 30.0
                        if abs(loss) > high_loss:
                            log.warning(
                                "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                                to_scalar(loss),
                                to_scalar(loss_dict['policy_loss']),
                                to_scalar(loss_dict['value_loss']),
                                to_scalar(loss_dict['exploration_loss']),
                                to_scalar(loss_dict['kl_loss']),
                            )

                            # perhaps something weird is happening, we definitely want summaries from this step
                            force_summaries = True
                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals()}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars), loss_dict)
                        del summary_vars
                        # TODO: Check how important this is / whether we want to use it
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

    def _record_summaries(self, train_loop_vars, losses) -> AttrDict:
        # FIXME: IMPROVISED FOR LOGGING
        self.last_summary_time = time.time()
        stats = AttrDict(losses)

        # TODO: Use 'new' way of logging (see below)

        ##########################################################################
        # log = FigureLogger(
        #     cfg.use_wandb, cfg.plot_dir, cfg.checkpoint_plot_dir, cfg.run_dir
        # )

        # log.plot_and_save_histories(histories)
        ##########################################################################

        # stats.lr = self.curr_lr
        # stats.actual_lr = train_loop_vars.actual_lr  # potentially scaled because of masked data

        # stats.update(self.actor_critic.summaries())  # TODO: Check actor_critic usage

        # # stats.valids_fraction = train_loop_vars.mb.valids.float().mean()
        # # stats.same_policy_fraction = (train_loop_vars.mb.policy_id == self.policy_id).float().mean()

        # grad_norm = (
        #     sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic.parameters() if p.grad is not None) ** 0.5
        # )
        # stats.grad_norm = grad_norm
        # stats.loss = train_loop_vars.loss
        # # stats.value = train_loop_vars.values.mean()
        # # stats.entropy = train_loop_vars.action_distribution.entropy().mean()
        # # stats.policy_loss = train_loop_vars.policy_loss
        # # stats.kl_loss = train_loop_vars.kl_loss
        # # stats.value_loss = train_loop_vars.value_loss
        # # stats.exploration_loss = train_loop_vars.exploration_loss

        # # stats.act_min = train_loop_vars.mb.actions.min()
        # # stats.act_max = train_loop_vars.mb.actions.max()

        # # if "adv_mean" in stats:
        # #     stats.adv_min = train_loop_vars.mb.advantages.min()
        # #     stats.adv_max = train_loop_vars.mb.advantages.max()
        # #     stats.adv_std = train_loop_vars.adv_std
        # #     stats.adv_mean = train_loop_vars.adv_mean

        # # stats.max_abs_logprob = torch.abs(train_loop_vars.mb.action_logits).max()

        # # if hasattr(train_loop_vars.action_distribution, "summaries"):
        # #     stats.update(train_loop_vars.action_distribution.summaries())

        # if train_loop_vars.epoch == self.cfg.num_epochs - 1 and train_loop_vars.batch_num == len(train_loop_vars.minibatches) - 1:
        #     # we collect these stats only for the last PPO batch, or every time if we're only doing one batch, IMPALA-style
        #     # valid_ratios = masked_select(train_loop_vars.ratio, train_loop_vars.mb.valids, train_loop_vars.num_invalids)
        #     # ratio_mean = torch.abs(1.0 - valid_ratios).mean().detach()
        #     # ratio_min = valid_ratios.min().detach()
        #     # ratio_max = valid_ratios.max().detach()
        #     # log.debug('Learner %d ratio mean min max %.4f %.4f %.4f', self.policy_id, ratio_mean.cpu().item(), ratio_min.cpu().item(), ratio_max.cpu().item())

        #     # value_delta = torch.abs(train_loop_vars.values - train_loop_vars.mb.values)
        #     # value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

        #     # stats.kl_divergence = train_loop_vars.kl_old_mean
        #     # stats.kl_divergence_max = train_loop_vars.kl_old.max()
        #     # stats.value_delta = value_delta_avg
        #     # stats.value_delta_max = value_delta_max
        #     # noinspection PyUnresolvedReferences
        #     # stats.fraction_clipped = (
        #     #     (valid_ratios < train_loop_vars.clip_ratio_low).float() + (valid_ratios > train_loop_vars.clip_ratio_high).float()
        #     # ).mean()
        #     # stats.ratio_mean = ratio_mean
        #     # stats.ratio_min = ratio_min
        #     # stats.ratio_max = ratio_max
        #     stats.num_sgd_steps = train_loop_vars.num_sgd_steps

        # # this caused numerical issues on some versions of PyTorch with second moment reaching infinity
        # adam_max_second_moment = 0.0
        # for key, tensor_state in self.optimizer.state.items():
        #     if "exp_avg_sq" in tensor_state:
        #         adam_max_second_moment = max(tensor_state["exp_avg_sq"].max().item(), adam_max_second_moment)
        # stats.adam_max_second_moment = adam_max_second_moment

        # version_diff = (train_loop_vars.curr_policy_version - train_loop_vars.mb.policy_version)[train_loop_vars.mb.policy_id == self.policy_id]
        # stats.version_diff_avg = version_diff.mean()
        # stats.version_diff_min = version_diff.min()
        # stats.version_diff_max = version_diff.max()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def _prepare_and_normalize_obs(self, obs: TensorDict) -> TensorDict:
        og_shape = dict()

        # assuming obs is a flat dict, collapse time and envs dimensions into a single batch dimension
        for key, x in obs.items():
            og_shape[key] = x.shape
            obs[key] = x.view((x.shape[0] * x.shape[1],) + x.shape[2:])

        # hold the lock while we alter the state of the normalizer since they can be used in other processes too
        with self.param_server.policy_lock:
            normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)  # TODO: Check actor_critic usage

        # restore original shape
        for key, x in normalized_obs.items():
            normalized_obs[key] = x.view(og_shape[key])

        return normalized_obs

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = shallow_recursive_copy(batch)

            # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
            valids: Tensor = buff["policy_id"] == self.policy_id
            # ignore experience that was older than the threshold even before training started
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            # for last T+1 step, we want to use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]

            # ensure we're in train mode so that normalization statistics are updated
            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # don't need non-normalized obs anymore

            # calculate estimated value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            next_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)["values"]
            # TODO: Check actor_critic usage
            buff["values"][:, -1] = next_values

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                # rl_games PPO uses a similar approach, see:
                # https://github.com/Denys88/rl_games/blob/7b5f9500ee65ae0832a7d8613b019c333ecd932c/rl_games/algos_torch/models.py#L51
                denormalized_values = buff["values"].clone()  # need to clone since normalizer is in-place
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
                # TODO: Check actor_critic usage
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values = buff["values"]

            if self.cfg.value_bootstrap:
                # Value bootstrapping is a technique that reduces the surprise for the critic in case
                # we're ending the episode by timeout. Intuitively, in this case the cumulative return for the last step
                # should not be zero, but rather what the critic expects. This improves learning in many envs
                # because otherwise the critic cannot predict the abrupt change in rewards in a timed-out episode.
                # What we really want here is v(t+1) which we don't have because we don't have obs(t+1) (since
                # the episode ended). Using v(t) is an approximation that requires that rew(t) can be generally ignored.

                # Multiply by both time_out and done flags to make sure we count only timeouts in terminal states.
                # There was a bug in older versions of isaacgym where timeouts were reported for non-terminal states.
                buff["rewards"].add_(self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"])

            if not self.cfg.with_vtrace:
                # calculate advantage estimate (in case of V-trace it is done separately for each minibatch)
                buff["advantages"] = gae_advantages(
                    buff["rewards"],
                    buff["dones"],
                    denormalized_values,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )
                # here returns are not normalized yet, so we should use denormalized values
                buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]

            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            # return normalization parameters are only used on the learner, no need to lock the mutex
            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])  # in-place # TODO: Check actor_critic usage

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(f"{self.policy_id=} batch has {invalid_fraction:.2%} of invalid samples")

                # invalid action values can cause problems when we calculate logprobs
                # here we set them to 0 just to be safe
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                # likewise, some invalid values of log_prob_actions can cause NaNs or infs
                buff["log_prob_actions"][invalid_indices] = -1  # -1 seems like a safe value

            return buff, dataset_size, num_invalids

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            buff, experience_size, num_invalids = self._prepare_batch(batch)

        if num_invalids >= experience_size:
            if self.cfg.with_pbt:
                log.warning("No valid samples in the batch, with PBT this must mean we just replaced weights")
            else:
                log.error(f"Learner {self.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self.timing.add_time("train"):
                train_stats = self._train(buff, self.cfg.batch_size, experience_size, num_invalids)

            # multiply the number of samples by frameskip so that FPS metrics reflect the number
            # of environment steps actually simulated
            if self.cfg.summaries_use_frameskip:
                self.env_steps += experience_size * self.env_info.frameskip
            else:
                self.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
            if train_stats is not None:
                if train_stats is not None:
                    stats[TRAIN_STATS] = train_stats
                stats[STATS_KEY] = memory_stats("learner", self.device)

            return stats

# TODO: Define save function using runner.save_checkpoint
