import argparse
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import (
    load_state_dict,
    make_env,
)
from sample_factory.huggingface.huggingface_utils import (
    generate_replay_video,
)
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import log
from tqdm import tqdm

from retinal_rl.analysis.activity_recording import (
    full_plot as extended_activity_raster_plot,
)
from retinal_rl.analysis.activity_recording import (
    sort_activity,
)
from retinal_rl.analysis.attribution import analyze as attribution_analyze
from retinal_rl.analysis.output_pca import analyze as output_pca_analyze
from retinal_rl.plot.util import fig_to_rgb_image
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from retinal_rl.util import rescale_zero_one
from runner.frameworks.rl.sf_framework import SFFramework

OmegaConf.register_new_resolver("eval", eval)


class VideoType(str, Enum):
    RAW = "RAW"
    AUGMENTED = "AUGMENTED"
    DECODED = "DECODED"
    VALUE_MASK = "VALUE_MASK"
    OUTPUT_PCA = "OUTPUT_PCA"
    ACTIVITY_RASTER = "ACTIVITY_RASTER"
    FULL = "FULL"


def video_type(value: str) -> VideoType:
    try:
        return VideoType(value)
    except ValueError:
        allowed = ", ".join([e.value for e in VideoType])
        raise argparse.ArgumentTypeError(
            f"invalid choice: {value!r} (choose from: {allowed})"
        )


def parse_args(argv: list[str] | None = None) -> tuple[Path, list[VideoType], bool]:
    parser = argparse.ArgumentParser(
        description="Select zero or more VideoTypes and a boolean flag."
    )

    parser.add_argument(
        "-e", "--experiment_path", type=Path, help="Path to the experiment directory."
    )

    parser.add_argument(
        "-t",
        "--type",
        metavar="VIDTYPE",
        type=video_type,
        nargs="+",
        default=["RAW"],
        help="Zero or more video types. Allowed: "
        + ", ".join([e.value for e in VideoType]),
    )

    # Single boolean flag: present -> True, absent -> False
    parser.add_argument(
        "--actor_frame_rate",
        action="store_true",
        help="Produce videos at the frame rate the actor operates at (will display only the frames the actor actually sees, typically 1/4 of the original frame rate).",
    )

    parser_args = parser.parse_args(argv)
    return parser_args.experiment_path, parser_args.type, parser_args.actor_frame_rate


def get_checkpoint_name(experiment_cfg) -> str:
    policy_id = experiment_cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[
        experiment_cfg.load_checkpoint_kind
    ]
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(experiment_cfg, policy_id), f"{name_prefix}_*"
    )
    return checkpoints[-1]


def create_video(
    experiment_path: Path, video_types: list[VideoType], actor_frame_rate: bool
):
    # Load the config file
    experiment_cfg = OmegaConf.load(experiment_path / "config" / "config.yaml")
    experiment_cfg.path.run_dir = experiment_path

    experiment_cfg.logging.use_wandb = False

    framework = SFFramework(experiment_cfg, "cache")
    run_simulation(framework.sf_cfg, video_types, actor_frame_rate)


def get_frames(
    actor_critic: SampleFactoryBrain, normalized_obs, rnn_states
) -> dict[VideoType, Any]:
    responses = actor_critic.brain(
        {"vision": normalized_obs["obs"], "rnn_state": rnn_states}
    )

    # find if decoder exists by matching output shape to input shape
    # TODO: Use loss definition instead and pass the key to the function
    decoder_key = None
    for response_key, response in responses.items():
        if (
            response[0].shape == normalized_obs["obs"].shape
            and response_key != "vision"
        ):
            decoder_key = response_key
            break

    cur_frames: dict[VideoType, Any] = {
        VideoType.AUGMENTED: normalized_obs["obs"].detach().cpu(),
        VideoType.DECODED: responses[decoder_key][0].detach().cpu()
        if decoder_key is not None
        else None,
        VideoType.VALUE_MASK: attribution_analyze(
            actor_critic.brain,
            {"vision": normalized_obs["obs"], "rnn_state": rnn_states},
            target_circuit="critic",
            method="l1",
            sum_channels=True,
            rescale_per_frame=True,
        )["vision"],
        VideoType.ACTIVITY_RASTER: {
            key: responses[key][0].flatten().detach().cpu()
            for key in responses
            if key != "vision"
        },
    }

    pca_frames = output_pca_analyze(
        responses,
        num_pcs=3,  # make rgb
        rescale_per_frame=False,
        circuit_names=["retina"],
    )

    cur_frames[VideoType.OUTPUT_PCA] = pca_frames["retina"]
    return cur_frames


def video_data_to_frames(
    video_data: dict[VideoType, Any],
    requested_types: list[VideoType],
) -> dict[VideoType, torch.Tensor]:
    frames: dict[VideoType, torch.Tensor] = {}

    # make activity data a dict of activity matrices
    activity_dict: dict[str, torch.Tensor] = {
        key: torch.vstack(
            [
                video_data[VideoType.ACTIVITY_RASTER][i][key]
                for i in range(len(video_data[VideoType.ACTIVITY_RASTER]))
            ]
        )
        for key in video_data[VideoType.ACTIVITY_RASTER][0]
    }
    sorted_activity: dict[str, torch.Tensor] = {
        key: sort_activity(activity_dict[key], "pca")[0] for key in ["rnn"]
    }  # we only want to plot rnn atm

    fig, ax = None, None
    for vid_type in requested_types:
        vid_data = video_data[vid_type]
        if vid_type == VideoType.ACTIVITY_RASTER:
            activity_frames = []
            for cur_frame in tqdm(
                range(len(video_data[VideoType.RAW])),
                "Processing frames for ACTIVITY_RASTER video...",
            ):
                fig, ax = extended_activity_raster_plot(
                    {
                        "vision": video_data[VideoType.RAW][cur_frame],
                        "rnn_state": None,
                    },
                    sorted_activity,
                    flatten_activity=False,
                    cur_frame=cur_frame,
                    num_frames=len(vid_data),
                    figure=(fig, ax) if fig else None,
                )

                torch_frame = fig_to_rgb_image(fig).swapaxes(2, 0).swapaxes(2, 1)[None]
                activity_frames.append(torch_frame)
            frames[vid_type] = np.concatenate(activity_frames, axis=0)
        elif vid_type == VideoType.FULL:
            additional_videos = [
                VideoType.DECODED,
                VideoType.AUGMENTED,
                VideoType.OUTPUT_PCA,
                VideoType.VALUE_MASK,
            ]
            activity_frames = []
            for cur_frame in tqdm(
                range(len(video_data[VideoType.RAW])),
                "Processing frames for FULL video...",
            ):
                fig, ax = extended_activity_raster_plot(
                    {
                        "vision": video_data[VideoType.RAW][cur_frame],
                        "rnn_state": None,
                    },
                    sorted_activity,
                    additional_images=[
                        video_data[_t][cur_frame] for _t in additional_videos
                    ],
                    additional_titles=[
                        "Reconstruction",
                        "Normalized Input",
                        "PCA of Retina Output",
                        "Value Attribution",
                    ],
                    cur_frame=cur_frame,
                    num_frames=len(video_data[VideoType.RAW]),
                    figure=(fig, ax) if fig else None,
                )

                torch_frame = fig_to_rgb_image(fig).swapaxes(2, 0).swapaxes(2, 1)[None]
                activity_frames.append(torch_frame)
            frames[vid_type] = np.concatenate(activity_frames, axis=0)
        else:
            all_frames = torch.cat(vid_data, dim=0)
            frames[vid_type] = (
                all_frames.movedim(1, -1).cpu().numpy()
            )  # move color channel to the end for video generation
    return frames


def run_simulation(  # noqa: C901 # TODO: Properly implement this anyway
    experiment_cfg: Config, video_types: list[VideoType], actor_frame_rate: bool
) -> StatusCode:
    experiment_cfg = load_from_checkpoint(experiment_cfg)

    eval_env_frameskip: int = (
        experiment_cfg.env_frameskip
        if experiment_cfg.eval_env_frameskip is None
        else experiment_cfg.eval_env_frameskip
    )
    assert (
        experiment_cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{experiment_cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = experiment_cfg.env_frameskip // eval_env_frameskip
    experiment_cfg.env_frameskip = experiment_cfg.eval_env_frameskip = (
        eval_env_frameskip
    )
    log.debug(
        f"Using frameskip {experiment_cfg.env_frameskip} and {render_action_repeat=} for evaluation"
    )

    experiment_cfg.num_envs = 1

    env = make_env(experiment_cfg, render_mode="rgb_array")
    env_info = extract_env_info(env, experiment_cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(
        experiment_cfg, env.observation_space, env.action_space
    )
    actor_critic.eval()

    device = torch.device("cpu" if experiment_cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    load_state_dict(experiment_cfg, actor_critic, device)

    num_frames = (
        1000 if experiment_cfg.max_num_frames is None else experiment_cfg.max_num_frames
    )

    video_frames = {vid_type: [] for vid_type in list(VideoType.__members__.keys())}
    terminated = True
    cur_frames = {}
    with torch.no_grad():
        for frame_no in tqdm(range(num_frames), "Generating video frames..."):
            if terminated:
                obs, _ = env.reset()
                action_mask = (
                    obs.pop("action_mask").to(device) if "action_mask" in obs else None
                )
                rnn_states = torch.zeros(
                    [env.num_agents, get_rnn_size(experiment_cfg)],
                    dtype=torch.float32,
                    device=device,
                )

            cur_frames[VideoType.RAW] = obs["obs"].detach().cpu()
            if frame_no % render_action_repeat == 0:
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                policy_outputs = actor_critic(
                    normalized_obs, rnn_states, action_mask=action_mask
                )

                actions = policy_outputs["actions"]

                if experiment_cfg.eval_deterministic:
                    action_distribution = actor_critic.action_distribution()
                    actions = argmax_actions(action_distribution)

                # actions shape should be [num_agents, num_actions] even if it's [1, 1]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)
                action_mask = (
                    obs.pop("action_mask").to(device) if "action_mask" in obs else None
                )
                rnn_states = policy_outputs["new_rnn_states"]
                cur_frames.update(get_frames(actor_critic, normalized_obs, rnn_states))

            if not actor_frame_rate:
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                cur_frames.update(get_frames(actor_critic, normalized_obs, rnn_states))

            for _vid_type in cur_frames:
                video_frames[_vid_type].append(cur_frames[_vid_type])

            obs, rew, terminated, truncated, infos = env.step(actions)

    env.close()

    fps = experiment_cfg.fps if experiment_cfg.fps > 0 else 30

    video_frames = video_data_to_frames(video_frames, requested_types=video_types)

    for _vid_type in video_types:
        # assert frames are in the right range (0-255) to produce the video
        shape = video_frames[_vid_type][0].shape
        for i, frame in enumerate(video_frames[_vid_type]):
            if frame.shape != shape:
                video_frames[_vid_type][i] = np.zeros(shape, dtype=np.uint8)
        video_frames[_vid_type] = (
            rescale_zero_one(np.stack(video_frames[_vid_type])) * 255
        ).astype(np.uint8)
        vid_path = experiment_path / "data" / "video"
        vid_path.mkdir(parents=True, exist_ok=True)

        ckpt_str = Path(get_checkpoint_name(experiment_cfg)).name[:-4]
        vid_type_str = f"_{_vid_type.value}"
        frame_rate_str = "_actor_frame_rate" if actor_frame_rate else ""
        experiment_cfg.video_name = ckpt_str + vid_type_str + frame_rate_str + ".mp4"
        generate_replay_video(
            str(vid_path), video_frames[_vid_type], fps, experiment_cfg
        )

    return ExperimentStatus.SUCCESS


if __name__ == "__main__":
    experiment_path, video_types, actor_frame_rate = parse_args()
    create_video(experiment_path, video_types, actor_frame_rate)
