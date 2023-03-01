### Util for preparing simulations and data for analysis

import numpy as np
import torch

from os.path import join

from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir

def analysis_path(cfg,nstps):

    return join(experiment_dir(cfg),"analyses",f"env_steps-{nstps}")

def data_path(cfg,nstps,flnm=None):

    datpth = analysis_path(cfg,nstps) + "/data"

    if flnm is not None:
        datpth = datpth + "/" + flnm

    return datpth

def plot_path(cfg,nstps,flnm=None):

    pltpth = analysis_path(cfg,nstps) + "/plots"

    if flnm is not None:
        pltpth = pltpth + "/" + flnm

    return pltpth


def save_onxx(cfg: Config, nstps : int, actor_critic : ActorCritic, env : BatchedVecEnv) -> None:
    """
    Write an onxx file of the saved model.
    """

    obs = env.observation_space.sample()
    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
    enc = actor_critic.encoder.basic_encoder
    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent

    # Note that onnx can't process dictionary inputs and so we can only look at the encoder (and decoder?) separately)
    torch.onnx.export(enc,(obs,),data_path(cfg,nstps,"encoder.onnx"),verbose=False,input_names=["observation"],output_names=["latent_state"])

def save_data(cfg : Config,nstps,dat,flnm):
    """
    Saves data. 'dat' should probably be a dictionary.
    """
    np.save(data_path(cfg,nstps,flnm), dat, allow_pickle=True)

def load_data(cfg : Config,nstps,flnm):
    """
    Loads data. Note the use of tolist() is necessary to read dictionaries.
    """
    return np.load(data_path(cfg,nstps,flnm) + ".npy", allow_pickle=True).tolist()
