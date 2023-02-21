### Util for preparing simulations and data for analysis

import numpy as np
import torch

from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir

def analysis_path(cfg,flnm=None):

    anapth = experiment_dir(cfg) + "/analysis"

    if flnm is not None:
        anapth = anapth + "/" + flnm

    return anapth


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
    torch.onnx.export(enc,obs,analysis_path(cfg,"encoder.onnx"),verbose=False,input_names=["observation"],output_names=["latent_state"])

def load_simulation(cfg : Config):
    return np.load(analysis_path(cfg,"simulation_recordings.npy"), allow_pickle=True).tolist()
