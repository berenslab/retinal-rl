### Util for preparing simulations and data for analysis

from typing import Tuple

import os
import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.make_env import make_env_func_batched,BatchedVecEnv
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic,ActorCritic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir, log
from sample_factory.algo.utils.env_info import extract_env_info

def get_ac_env(cfg: Config) -> Tuple[ActorCritic, BatchedVecEnv]:
    """
    Load the model from checkpoint, initialize the environment, and return both.
    """
    #verbose = False

    cfg = load_from_checkpoint(cfg)

    cfg.env_frameskip = cfg.eval_env_frameskip

    cfg.num_envs = 1

    # In general we only focus on saving to files
    render_mode = "rgb_array"

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    return actor_critic,env

def save_onxx(cfg: Config, actor_critic : ActorCritic, env : BatchedVecEnv) -> None:
    """
    Write an onxx file of the saved model.
    """

    obs, _ = env.reset()
    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
    enc = actor_critic.encoder.basic_encoder
    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]

    # Note that onnx can't process dictionary inputs and so we can only look at the encoder (and decoder?) separately)
    torch.onnx.export(enc,obs,experiment_dir(cfg) + "/encoder.onnx",verbose=False,input_names=["observation"],output_names=["latent_state"])

def normalized_obs_to_img(normalized_obs):
    """
    Rearrange an image so it can be presented by matplot lib.
    """
    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    img = obs.cpu().numpy()
    return img

def save_simulation_data(cfg: Config, actor_critic : ActorCritic, env : BatchedVecEnv) -> None:
    """
    Save an example simulation.
    """

    # Initializing some local variables
    t_max = int(cfg.analyze_max_num_frames)
    env_info = extract_env_info(env, cfg)
    action_repeat: int = cfg.env_frameskip // cfg.eval_env_frameskip
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    # Initializing stream arrays
    all_img = np.zeros((cfg.res_h, cfg.res_w, 3, t_max)).astype(np.uint8)
    all_rnn_act = np.zeros((cfg.rnn_size, t_max))
    all_actions = np.zeros((2, t_max))
    all_health = np.zeros(t_max)

    # Initializing simulation state
    num_frames = 0
    num_episodes = 0
    obs,_ = env.reset()
    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    # Simulation loop
    with torch.no_grad():

        while t_max > num_frames:

            # Evaluate policy
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            rnn_states = policy_outputs["new_rnn_states"]
            actions = policy_outputs["actions"]

            # Prepare network state for saving
            rnn_act = rnn_states.cpu().detach().numpy()

            # can pass --eval_deterministic=True to CLI in order to argmax the probabilistic actions
            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            # Repeating actions during evaluation because we run the simulation at higher FPS
            for _ in range(action_repeat):

                img = normalized_obs_to_img(normalized_obs).astype(np.uint8) # presumeably this second type call is redundant
                health = env.unwrapped.get_info()['HEALTH'] # environment info (health etc.)

                all_img[:,:,:,num_frames] = img
                all_rnn_act[:,num_frames] = rnn_act
                all_actions[:,num_frames] = actions
                all_health[num_frames] = health

                obs,rew,_,_,_ = env.step(actions)
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                actions = np.array(actions)
                health = env.unwrapped.get_info()['HEALTH'] # environment info (health etc.)

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

    analyze_out = {
            'all_img': all_img,
            'all_rnn_act':all_rnn_act,
            'all_actions':all_actions,
            'all_health':all_health,
            }

    np.save(f'{os.getcwd()}/train_dir/{cfg.experiment}/analyze_out.npy', analyze_out, allow_pickle=True)

