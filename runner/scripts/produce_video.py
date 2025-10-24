import argparse
from enum import Enum
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
import torch
from omegaconf import OmegaConf
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import (
    load_state_dict,
    make_env,
    render_frame,
    visualize_policy_inputs,
)
from sample_factory.huggingface.huggingface_utils import (
    generate_model_card,
    generate_replay_video,
    push_to_hf,
)
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import experiment_dir, log
from sample_factory.algo.learning.learner import Learner

from runner.frameworks.rl.sf_framework import SFFramework

OmegaConf.register_new_resolver("eval", eval)

class VideoType(str, Enum):
    RAW = "RAW"
    AUGMENTED = "AUGMENTED"
    DECODED = "DECODED"
    VALUE_MASK = "VALUE_MASK"

def video_type(value: str) -> VideoType:
    try:
        return VideoType(value)
    except ValueError:
        allowed = ", ".join([e.value for e in VideoType])
        raise argparse.ArgumentTypeError(f"invalid choice: {value!r} (choose from: {allowed})")

def parse_args(argv: list[str] | None = None) -> tuple[Path, list[VideoType], bool]:
    parser = argparse.ArgumentParser(
        description="Select zero or more VideoTypes and a boolean flag."
    )

    parser.add_argument(
        "-e", "--experiment_path",
        type=Path,
        help="Path to the experiment directory."
    )

    parser.add_argument(
        "-t", "--type",
        metavar="VIDTYPE",
        type=video_type,
        nargs="+",
        default=["RAW"],
        help="Zero or more video types. Allowed: " + ", ".join([e.value for e in VideoType])
    )

    # Single boolean flag: present -> True, absent -> False
    parser.add_argument(
        "--actor_frame_rate",
        action="store_true",
        help="Produce videos at the frame rate the actor operates at (will display only the frames the actor actually sees, typically 1/4 of the original frame rate)."
    )

    parser_args = parser.parse_args(argv)
    return parser_args.experiment_path, parser_args.type, parser_args.actor_frame_rate

def get_checkpoint_name(experiment_cfg) -> str:
    policy_id = experiment_cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[experiment_cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(experiment_cfg, policy_id), f"{name_prefix}_*")
    return checkpoints[-1]

def create_video(experiment_path: Path, video_types: list[VideoType], actor_frame_rate: bool):
    # Load the config file
    experiment_cfg = OmegaConf.load(experiment_path / "config" / "config.yaml")
    experiment_cfg.path.run_dir = experiment_path

    experiment_cfg.logging.use_wandb = False

    framework = SFFramework(experiment_cfg, "cache")
    custom_enjoy(framework.sf_cfg, video_types, actor_frame_rate)


def _rescale_zero_one(x, min: Optional[float] = None, max: Optional[float] = None):
    if min is None:
        min = np.min(x)
    if max is None:
        max = np.max(x)
    return (x - min) / (max - min)


def get_frames(actor_critic: SampleFactoryBrain, obs, rnn_states) -> dict[VideoType, torch.Tensor]:
    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

    responses = actor_critic.brain(
        {"vision": normalized_obs["obs"], "rnn_state": rnn_states}
    )

    cur_frames: dict[VideoType, torch.Tensor] = {
        VideoType.RAW: obs["obs"],
        VideoType.AUGMENTED: normalized_obs["obs"],
        VideoType.DECODED: responses.get("v1_decoder", None), # TODO: automatically detect decoder name
        VideoType.VALUE_MASK: None,
    }
    return cur_frames

def custom_enjoy(  # noqa: C901 # TODO: Properly implement this anyway
    experiment_cfg: Config,
    video_types: list[VideoType],
    actor_frame_rate: bool,
) -> tuple[StatusCode, float]:
    verbose = False

    experiment_cfg = load_from_checkpoint(experiment_cfg)

    eval_env_frameskip: int = (
        experiment_cfg.env_frameskip if experiment_cfg.eval_env_frameskip is None else experiment_cfg.eval_env_frameskip
    )
    assert (
        experiment_cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{experiment_cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = experiment_cfg.env_frameskip // eval_env_frameskip
    experiment_cfg.env_frameskip = experiment_cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(
        f"Using frameskip {experiment_cfg.env_frameskip} and {render_action_repeat=} for evaluation"
    )

    experiment_cfg.num_envs = 1

    render_mode = "rgb_array"

    env = make_env(experiment_cfg, render_mode=render_mode)
    env_info = extract_env_info(env, experiment_cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(experiment_cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if experiment_cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    load_state_dict(experiment_cfg, actor_critic, device)

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames: int) -> bool:
        return experiment_cfg.max_num_frames is not None and frames > experiment_cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros(
        [env.num_agents, get_rnn_size(experiment_cfg)], dtype=torch.float32, device=device
    )
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = {vid_type: [] for vid_type in video_types}
    num_episodes = 0

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            cur_frames: dict[VideoType, torch.Tensor] = get_frames(actor_critic, obs, rnn_states)
            policy_outputs = actor_critic(
                normalized_obs, rnn_states, action_mask=action_mask
            )

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if experiment_cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _i_repeat in range(render_action_repeat):
                obs, rew, terminated, truncated, infos = env.step(actions)

                need_video_frame = (
                    len(next(iter(video_frames.items()))) < experiment_cfg.video_frames
                    or experiment_cfg.video_frames < 0
                    and num_episodes == 0
                )
                if need_video_frame:
                    # frame = env.render()
                    if not actor_frame_rate and _i_repeat > 0:
                        cur_frames = get_frames(actor_critic, obs, rnn_states)

                    for _vid_type in video_types:
                        video_frames[_vid_type].append(cur_frames[_vid_type][0].movedim(0, -1).cpu().numpy())

                action_mask = (
                    obs.pop("action_mask").to(device) if "action_mask" in obs else None
                )
                dones = make_dones(terminated, truncated)
                infos = (
                    [{} for _ in range(env_info.num_agents)] if infos is None else infos
                )

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                true_objectives[agent_i][-1],
                            )
                        rnn_states[agent_i] = torch.zeros(
                            [get_rnn_size(experiment_cfg)], dtype=torch.float32, device=device
                        )
                        episode_reward[agent_i] = 0

                        if experiment_cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i]:
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    render_frame(
                        experiment_cfg, env, video_frames[VideoType.RAW], num_episodes, last_render_start
                    )
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s",
                        avg_episode_rewards_str,
                        avg_true_objective_str,
                    )
                    log.info(
                        "Avg episode reward: %.3f, avg true_objective: %.3f",
                        np.mean(
                            [np.mean(episode_rewards[i]) for i in range(env.num_agents)]
                        ),
                        np.mean(
                            [np.mean(true_objectives[i]) for i in range(env.num_agents)]
                        ),
                    )

            if num_episodes >= experiment_cfg.max_num_episodes:
                break

    env.close()

    fps = experiment_cfg.fps if experiment_cfg.fps > 0 else 30

    for _vid_type in video_types:
        # assert frames are in the right range (0-255) to produce the video
        shape = video_frames[_vid_type][0].shape
        for i, frame in enumerate(video_frames[_vid_type]):
            if frame.shape != shape:
                video_frames[_vid_type][i] = np.zeros(shape, dtype=np.uint8)
        video_frames[_vid_type] = (_rescale_zero_one(np.stack(video_frames[_vid_type])) * 255).astype(
            np.uint8
        )
        vid_path = experiment_path / "data" / "video"
        vid_path.mkdir(parents=True, exist_ok=True)

        ckpt_str = Path(get_checkpoint_name(experiment_cfg)).name[:-4]
        vid_type_str = f"_{_vid_type.value}"
        frame_rate_str = "_actor_frame_rate" if actor_frame_rate else ""
        experiment_cfg.video_name = ckpt_str + vid_type_str + frame_rate_str + ".mp4"
        generate_replay_video(
            str(vid_path), video_frames[_vid_type], fps, experiment_cfg
        )

    return ExperimentStatus.SUCCESS, sum(
        [sum(episode_rewards[i]) for i in range(env.num_agents)]
    ) / max(1, sum([len(episode_rewards[i]) for i in range(env.num_agents)]))


if __name__ == "__main__":
    experiment_path, video_types, actor_frame_rate = parse_args()
    create_video(experiment_path, video_types, actor_frame_rate)
