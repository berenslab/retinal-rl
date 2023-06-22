import os
from os.path import join
import functools
from typing import Optional
import numpy as np

from sample_factory.envs.env_utils import register_env

from sf_examples.vizdoom.doom.doom_utils import DoomSpec, make_doom_env_impl

import gym
from gym.spaces import Discrete



### Action Spaces ###


def key_to_action_basic(key):
    from pynput.keyboard import Key

    table = {Key.left: 0, Key.right: 1, Key.up: 2, Key.down: 3}
    return table.get(key, None)

def doom_action_space_basic():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    MOVE_BACKWARD
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(3),
        )
    )  # noop, turn left, turn right  # noop, forward, backward

    space.key_to_action = key_to_action_basic
    return space


### Wrappers ###

class HungerInput(gym.Wrapper):
    """Add game variables to the observation space + reward shaping."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        low = [-1.0]

        high = [1.0]

        self.observation_space = gym.spaces.Dict(
            {
                "obs": current_obs_space,
                "measurements": gym.spaces.Box(
                    low=np.array(low, dtype=np.float32),
                    high=np.array(high, dtype=np.float32),
                ),
            }
        )

        num_measurements = 1

        self.measurements_vec = np.zeros([num_measurements], dtype=np.float32)

    def _parse_info(self, obs, info):

        # we don't really care how much negative health we have, dead is dead
        hlth = float(info["HEALTH"])
        # clip health to [-1,1]
        hlth = np.clip(hlth, 0, 100)
        hunger = (50 - hlth) / 50.0
        self.measurements_vec[0] = hunger
        obs_dict = {"obs": obs, "measurements": self.measurements_vec}

        return obs_dict

    def reset(self, **kwargs):

        obs, _ = self.env.reset(**kwargs)
        info = self.env.unwrapped.get_info()
        obs = self._parse_info(obs, info)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, rew, terminated, truncated, info

        obs_dict = self._parse_info(obs, info)

        return obs_dict, rew, terminated, truncated, info


### Retinal Environments ###

def retinal_doomspec(scnr,flnm):
    return DoomSpec( scnr
                    , join(os.getcwd(), "scenarios", flnm)
                    , doom_action_space_basic()
                    , reward_scaling=1
                    , extra_wrappers= [(HungerInput, {})]
                    )

def make_retinal_env_from_spec(spec, _env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    """
    Makes a Retinal environment from a DoomSpec instance.
    """

    res = "{cfg.res_w}x{cfg.res_h}".format(cfg=cfg)

    return make_doom_env_impl(spec, cfg=cfg, env_config=env_config, render_mode=render_mode, custom_resolution=res, **kwargs)

def register_retinal_env(scnnm):

    cfgnm = scnnm + ".cfg"

    env_spec = retinal_doomspec(scnnm, cfgnm)
    make_env_func = functools.partial(make_retinal_env_from_spec, env_spec)
    register_env(env_spec.name, make_env_func)
