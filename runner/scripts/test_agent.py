import multiprocessing as mp
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import load_state_dict
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log

from retinal_rl.rl.sample_factory.environment import register_retinal_env
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from runner.frameworks.rl.sf_framework import SFFramework

"""
Allows to run a trained agent in an evironment that can be specified to test how well it survives in that.
Usage: python -m runner.scripts.test_agent {path_to_experiment} {env_name} {num_repeats}
Stores the results as a list in data/analyses/survival_durations_{env_name}.csv
"""


OmegaConf.register_new_resolver("eval", eval)


def _episode_worker(
    task_queue: "mp.Queue[object]",
    result_queue: "mp.Queue[int]",
    cfg: Config,
    data_root: str,
    worker_index: int,
) -> None:
    """
    Worker process: loads model once, then runs episodes pulled from task_queue.
    Sends frame counts to result_queue. Stops when it receives None from task_queue.
    """
    register_retinal_env(cfg.env, data_root, cfg.input_satiety, cfg.allow_backwards)
    global_model_factory().register_actor_critic_factory(SampleFactoryBrain)

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = (
        cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    )
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=worker_index, vector_index=0, env_id=worker_index),
    )
    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    env_info = extract_env_info(env, cfg)

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    load_state_dict(cfg, actor_critic, device)

    rnn_size = get_rnn_size(cfg)

    while True:
        task = task_queue.get()
        if task is None:
            break

        frame_count = 0
        obs, _ = env.reset()
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        rnn_states = torch.zeros([1, rnn_size], dtype=torch.float32, device=device)

        episode_done = False
        with torch.no_grad():
            while not episode_done:
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                policy_outputs = actor_critic(
                    normalized_obs, rnn_states, action_mask=action_mask
                )

                actions = policy_outputs["actions"]
                if cfg.eval_deterministic:
                    action_distribution = actor_critic.action_distribution()
                    actions = argmax_actions(action_distribution)

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
                    frame_count += 1
                    if torch.as_tensor(dones).any():
                        episode_done = True
                        break

        result_queue.put(frame_count)

    env.close()


def test_survival_duration(
    cfg: Config,
    num_repeats: int = 10,
    data_root: str = "cache",
) -> list[int]:
    batch_size = min(num_repeats, cfg.num_workers)
    log.debug(f"Running {num_repeats} episodes with {batch_size} parallel workers")

    task_queue: "mp.Queue[object]" = mp.Queue()
    result_queue: "mp.Queue[int]" = mp.Queue()

    for _ in range(num_repeats):
        task_queue.put(1)
    for _ in range(batch_size):
        task_queue.put(None)  # one stop signal per worker

    workers = [
        mp.Process(
            target=_episode_worker,
            args=(task_queue, result_queue, cfg, data_root, i),
            daemon=True,
        )
        for i in range(batch_size)
    ]
    for w in workers:
        w.start()

    results = [result_queue.get() for _ in range(num_repeats)]

    for w in workers:
        w.join()

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
        framework.sf_cfg, num_repeats=num_repeats, data_root=framework.data_root
    )

    duration_file = experiment_path / "data" / "analyses" / f"survival_durations_{env_name}.csv"
    if duration_file.exists():
        with open(duration_file, "r") as f:
            existing_durations = [int(line.strip()) for line in f]
        survival_durations = existing_durations + survival_durations

    with open(
        experiment_path / "data" / "analyses" / f"survival_durations_{env_name}.csv",
        "w",
    ) as f:
        for duration in survival_durations:
            f.write(f"{duration}\n")
