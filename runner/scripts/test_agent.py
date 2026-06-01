import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import EnvInfo, extract_env_info
from sample_factory.algo.utils.make_env import BatchedVecEnv, make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import load_state_dict
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log

from runner.frameworks.rl.sf_framework import SFFramework

"""
Allows to run a trained agent in an evironment that can be specified to test how well it survives in that.
Usage: python -m runner.scripts.test_agent {path_to_experiment} {env_name} {num_repeats}
Stores the results as a list in data/analyses/survival_durations_{env_name}.csv
"""


OmegaConf.register_new_resolver("eval", eval)


@dataclass
class _InferReq:
    worker_id: int
    obs: dict
    rnn_states: torch.Tensor
    action_mask: Optional[torch.Tensor]


def _inference_server(
    infer_queue: "queue.Queue[Optional[_InferReq]]",
    result_queues: "list[queue.Queue]",
    actor_critic: ActorCritic,
    cfg: Config,
) -> None:
    """
    Collects inference requests from all worker threads, runs one batched forward
    pass, and distributes results. This avoids N serial batch-size-1 passes.
    """
    while True:
        reqs: list[_InferReq] = []

        req = infer_queue.get()
        if req is None:
            break
        reqs.append(req)

        # Drain any additional pending requests without waiting
        while True:
            try:
                req = infer_queue.get_nowait()
                if req is None:
                    infer_queue.put(None)  # put back the stop signal
                    break
                reqs.append(req)
            except queue.Empty:
                break

        obs_keys = list(reqs[0].obs.keys())
        batch_obs = {k: torch.cat([r.obs[k] for r in reqs], dim=0) for k in obs_keys}
        batch_rnn = torch.cat([r.rnn_states for r in reqs], dim=0)
        has_mask = reqs[0].action_mask is not None
        batch_mask = torch.cat([r.action_mask for r in reqs], dim=0) if has_mask else None

        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(actor_critic, batch_obs)
            policy_outputs = actor_critic(normalized_obs, batch_rnn, action_mask=batch_mask)
            actions = policy_outputs["actions"]
            if cfg.eval_deterministic:
                actions = argmax_actions(actor_critic.action_distribution())
            new_rnn_states = policy_outputs["new_rnn_states"]

        for i, req in enumerate(reqs):
            result_queues[req.worker_id].put(
                {
                    "actions": actions[i : i + 1],
                    "new_rnn_states": new_rnn_states[i : i + 1],
                }
            )


def _episode_worker(
    task_queue: "queue.Queue[object]",
    episode_result_queue: "queue.Queue[int]",
    infer_queue: "queue.Queue[Optional[_InferReq]]",
    my_result_queue: "queue.Queue[dict]",
    env: BatchedVecEnv,
    cfg: Config,
    env_info: EnvInfo,
    render_action_repeat: int,
    device: torch.device,
    worker_id: int,
) -> None:
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
        while not episode_done:
            obs_device = {k: v.to(device) for k, v in obs.items()}
            infer_queue.put(
                _InferReq(
                    worker_id=worker_id,
                    obs=obs_device,
                    rnn_states=rnn_states,
                    action_mask=action_mask,
                )
            )

            result = my_result_queue.get()
            actions = result["actions"]
            rnn_states = result["new_rnn_states"]

            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

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

        episode_result_queue.put(frame_count)


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

    batch_size = min(num_repeats, cfg.num_workers)
    log.debug(f"Running {num_repeats} episodes with {batch_size} parallel threads")

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    envs = [
        make_env_func_batched(
            cfg,
            env_config=AttrDict(worker_index=0, vector_index=i, env_id=i),
        )
        for i in range(batch_size)
    ]
    for env in envs:
        if hasattr(env.unwrapped, "reset_on_init"):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False

    env_info = extract_env_info(envs[0], cfg)

    actor_critic = create_actor_critic(cfg, envs[0].observation_space, envs[0].action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    load_state_dict(cfg, actor_critic, device)

    task_queue: "queue.Queue[object]" = queue.Queue()
    episode_result_queue: "queue.Queue[int]" = queue.Queue()
    infer_queue: "queue.Queue[Optional[_InferReq]]" = queue.Queue()
    result_queues: "list[queue.Queue[dict]]" = [queue.Queue() for _ in range(batch_size)]

    for _ in range(num_repeats):
        task_queue.put(1)
    for _ in range(batch_size):
        task_queue.put(None)

    inference_thread = threading.Thread(
        target=_inference_server,
        args=(infer_queue, result_queues, actor_critic, cfg),
        daemon=True,
    )
    inference_thread.start()

    worker_threads = [
        threading.Thread(
            target=_episode_worker,
            args=(
                task_queue,
                episode_result_queue,
                infer_queue,
                result_queues[i],
                envs[i],
                cfg,
                env_info,
                render_action_repeat,
                device,
                i,
            ),
            daemon=True,
        )
        for i in range(batch_size)
    ]
    for t in worker_threads:
        t.start()

    results = []
    for i in range(num_repeats):
        frame_count = episode_result_queue.get()
        results.append(frame_count)
        log.info(f"Episode {i + 1}/{num_repeats} completed: {frame_count} frames")

    for t in worker_threads:
        t.join()

    infer_queue.put(None)
    inference_thread.join()

    for env in envs:
        env.close()

    return results


if __name__ == "__main__":
    experiment_path = Path(sys.argv[1])
    env_name = sys.argv[2]
    num_repeats = int(sys.argv[3])

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
        "a",
    ) as f:
        f.writelines([f"{duration}\n" for duration in survival_durations])
