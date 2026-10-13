### Util for preparing simulations and data for analysis

from typing import Tuple

import torch
from sample_factory.algo.utils.make_env import BatchedVecEnv, make_env_func_batched
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log

torch.backends.cudnn.enabled = False


def get_brain_env(
    cfg: Config, checkpoint_dict
) -> Tuple[ActorCritic, BatchedVecEnv, AttrDict, int]:
    """
    Load the model from checkpoint, initialize the environment, and return both.
    """
    # verbose = False

    cfg.env_frameskip = cfg.eval_env_frameskip

    cfg.num_envs = 1

    # In general we only focus on saving to files
    render_mode = "rgb_array"

    log.debug("RETINAL RL: Making environment...")
    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=render_mode,
    )

    log.debug("RETINAL RL: Finished making environment, loading actor-critic model...")
    brain = create_actor_critic(cfg, env.observation_space, env.action_space)
    # log.debug("RETINAL RL: ...evaluating actor-critic model...")
    brain.eval()

    # log.debug("RETINAL RL: Actor-critic initialized...")

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    brain.model_to_device(device)

    # log.debug("RETINAL RL: ...copied to device...")

    brain.load_state_dict(checkpoint_dict["model"])
    nstps = checkpoint_dict["env_steps"]

    # log.debug("RETINAL RL: ...and loaded from checkpoint.")

    return brain, env, cfg, nstps
