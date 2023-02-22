### Util for preparing simulations and data for analysis

import numpy as np
import torch

from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir

def analysis_path(cfg):

    return experiment_dir(cfg) + "/analyses"

def data_path(cfg,flnm=None):

    datpth = analysis_path(cfg) + "/data"

    if flnm is not None:
        datpth = datpth + "/" + flnm

    return datpth

def plot_path(cfg,flnm=None):

    pltpth = analysis_path(cfg) + "/plots"

    if flnm is not None:
        pltpth = pltpth + "/" + flnm

    return pltpth


def save_onxx(cfg: Config, actor_critic : ActorCritic, env : BatchedVecEnv) -> None:
    """
    Write an onxx file of the saved model.
    """

    obs, _ = env.reset()
    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
    enc = actor_critic.encoder.basic_encoder
    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent

    # Note that onnx can't process dictionary inputs and so we can only look at the encoder (and decoder?) separately)
    torch.onnx.export(enc,(obs,),data_path(cfg,"encoder.onnx"),verbose=False,input_names=["observation"],output_names=["latent_state"])

def save_data(cfg : Config,dat,flnm):
    np.save(data_path(cfg,flnm), dat, allow_pickle=True)

def load_data(cfg : Config,flnm):
    return np.load(data_path(cfg,flnm) + ".npy", allow_pickle=True).tolist()
