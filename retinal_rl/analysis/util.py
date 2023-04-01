### Util for preparing simulations and data for analysis

import numpy as np
import torch

import os
from os.path import join

from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir

from torch import nn


## Paths ###


def analysis_root(cfg):
    """
    Returns the root analysis directory.
    """

    return join(experiment_dir(cfg),"analyses")

def analysis_path(cfg,nstps):
    """
    Returns the path to the analysis directory.
    """
    art = analysis_root(cfg)

    return join(art,f"env_steps-{nstps}")

def get_analysis_times(cfg):
    """
    Returns the list of analysis times.
    """
    art = analysis_root(cfg)
    return [int(f.split("-")[1]) for f in os.listdir(art)]

def data_path(cfg,nstps,flnm=None):
    """
    Returns the path to the data directory.
    """

    datpth = analysis_path(cfg,nstps) + "/data"

    if flnm is not None:
        datpth = datpth + "/" + flnm

    return datpth

def plot_path(cfg,nstps,flnm=None):
    """
    Returns the path to the plot directory.
    """

    pltpth = analysis_path(cfg,nstps) + "/plots"

    if flnm is not None:
        pltpth = pltpth + "/" + flnm

    return pltpth


### IO ###


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
    torch.onnx.export(enc,torch.unsqueeze(obs,0),data_path(cfg,nstps,"encoder.onnx"),verbose=False,input_names=["observation"],output_names=["latent_state"])

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


### Misc analysis tools ###

def normalize_data(xs):
    return (xs - np.min(xs)) / (np.max(xs) - np.min(xs))

def from_float_to_rgb(xs):
    return (255*normalize_data(xs)).astype(np.uint8)

def obs_dict_to_obs(obs_dct):
    """
    Extract observation
    """
    obs = obs_dct["obs"]
    # visualize obs only for the 1st agent
    return obs[0]


def obs_to_img(obs):
    """
    Rearrange an image so it can be presented by matplot lib.
    """
    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    img = obs.cpu().numpy()
    return img


class ValueNetwork(nn.Module):

    """
    Converts a LindseyEncoder into a feedforward value network that can be easily analyzed by e.g. captum.
    """
    def __init__(self, cfg, actor_critic):

        super().__init__()

        self.cfg = cfg
        self.ac_base = actor_critic

        self.conv_head_out_size = actor_critic.encoder.basic_encoder.conv_head_out_size

        self.conv_head = actor_critic.encoder.basic_encoder.conv_head
        self.fc1 = actor_critic.encoder.basic_encoder.fc1 # here we will need to flatten the features before going forward
        self.nl_fc = actor_critic.encoder.basic_encoder.nl_fc

        self.critic = actor_critic.critic_linear


    def forward(self, nobs):
        # conv layer 1

        x = self.conv_head(nobs)
        x = x.contiguous().view(-1, self.conv_head_out_size)

        x = self.fc1(x)
        x = self.nl_fc(x)

        x = self.critic(x)

        return x


