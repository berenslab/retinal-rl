import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import (
    load_state_dict,
    make_env,
)
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log

from runner.frameworks.rl.sf_framework import SFFramework

"""
Allows to run a trained agent in an evironment that can be specified to test how well it survives in that.
Usage: python -m runner.scripts.test_agent {path_to_experiment} {env_name} {num_repeats}
Stores the results as a list in data/analyses/survival_durations_{env_name}.csv
"""


OmegaConf.register_new_resolver("eval", eval)


def test_survival_duration(
    cfg: Config,
    num_repeats: int = 10,
) -> list[int]:
    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = (
        cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    )
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(
        f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation"
    )

    batch_size = min(num_repeats, cfg.num_workers)
    cfg.num_envs = batch_size

    env = make_env(cfg)

    assert env.num_agents == 1, "env.num_agents must be 1"

    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    load_state_dict(cfg, actor_critic, device)

    results: list[int] = []
    frame_counts = [0] * batch_size  # frame count for the current episode per env

    obs, _ = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros(
        [batch_size, get_rnn_size(cfg)], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        while len(results) < num_repeats:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            policy_outputs = actor_critic(
                normalized_obs, rnn_states, action_mask=action_mask
            )

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                obs, _, terminated, truncated, _ = env.step(actions)

                action_mask = (
                    obs.pop("action_mask").to(device) if "action_mask" in obs else None
                )
                dones = make_dones(terminated, truncated)

                for i, done in enumerate(torch.as_tensor(dones).view(batch_size)):
                    frame_counts[i] += 1
                    if done:
                        results.append(frame_counts[i])
                        frame_counts[i] = 0
                        rnn_states[i] = torch.zeros(
                            [get_rnn_size(cfg)], dtype=torch.float32, device=device
                        )
                        if len(results) >= num_repeats:
                            break

                if len(results) >= num_repeats:
                    break

    env.close()
    return results


if __name__ == "__main__":
    experiment_path = Path(sys.argv[1])
    env_name = sys.argv[2]
    num_repeats = int(sys.argv[3])

    # Load the config file
    cfg = OmegaConf.load(experiment_path / "config" / "config.yaml")
    cfg.path.run_dir = experiment_path

    cfg.dataset.env_name = env_name

    cfg.logging.use_wandb = False
    cfg.samplefactory.no_render = True

    framework = SFFramework(cfg, "cache")
    survival_durations = test_survival_duration(
        framework.sf_cfg, num_repeats=num_repeats
    )

    with open(
        experiment_path / "data" / "analyses" / f"survival_durations_{env_name}.csv",
        "w",
    ) as f:
        for duration in survival_durations:
            f.write(f"{duration}\n")
