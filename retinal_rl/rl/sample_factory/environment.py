import functools
import os
from typing import Optional

import gymnasium as gym
import numpy as np

# import gym
from gymnasium.spaces import Discrete
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.action_space import (
    doom_action_space_basic,
    key_to_action_basic,
)
from sf_examples.vizdoom.doom.doom_utils import DoomSpec, make_doom_env_impl

# from gym.spaces import Discrete


### Action Spaces ###


def doom_action_space_no_backwards():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
        )
    )  # noop, turn left, turn right  # noop, forward

    space.key_to_action = key_to_action_basic
    return space


### Wrappers ###


class SatietyInput(gym.Wrapper):
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
        hlth = float(
            info["HEALTH"]
        )  # TODO: Used when input_satiety = true - but info does not contain HEALTH
        # clip health to [-1,1]
        hlth = np.clip(hlth, 0, 100)
        satiety = (hlth - 50) / 50.0
        self.measurements_vec[0] = satiety
        return {"obs": obs, "measurements": self.measurements_vec}

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


class PickupTrackingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # TODO: make object values configurable / get from environment
        self.object_values = np.array([-25, -20, -15, -10, -5, 10, 20, 30, 40, 50])
        self.pickup_counts = {str(object_value): 0 for object_value in self.object_values}
        self.last_health = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pickup_counts = {str(object_value): 0 for object_value in self.object_values}
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)

        unbound_health = info.get("USER17")
        diff = unbound_health - self.last_health if self.last_health is not None else 0
        self.last_health = unbound_health
        # Accept differences up to one (eg health decreases by 1 periodically, which can coincide with the pickup of an object)
        picked_up_object = np.abs(diff - self.object_values) <= 1
        if np.sum(picked_up_object) == 1:
            for object_value in self.object_values[picked_up_object]:
                self.pickup_counts[str(object_value)] += 1

        if picked_up_object.sum() > 1 or (
            picked_up_object.sum() == 0 and diff not in [0, -1]
        ):
            print(
                f"Warning: Detected pickup of multiple objects or an object with an unexpected value. Diff: {diff}, Picked up objects: {self.object_values[picked_up_object]}"
            )
            # TODO: proper logging

        if "episode_extra_stats" not in info:
            info["episode_extra_stats"] = dict()
        info["episode_extra_stats"]["pickup_counts"] = (
            self.pickup_counts.copy()
        )  # Add a copy of the pickup counts to the info dict

        return obs, rew, terminated, truncated, info


### Retinal Environments ###


def retinal_doomspec(
    scene_name: str, cfg_path: str, sat_in: bool, allow_backwards: bool
):
    ewraps = [(PickupTrackingWrapper, {})]

    if sat_in:
        ewraps.append(SatietyInput, {})

    action_space = (
        doom_action_space_basic()
        if allow_backwards
        else doom_action_space_no_backwards()
    )
    return DoomSpec(
        scene_name,
        cfg_path,
        action_space,
        reward_scaling=1,
        extra_wrappers=ewraps,
    )


def make_retinal_env_from_spec(
    spec, _env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs
):
    """
    Makes a Retinal environment from a DoomSpec instance.
    """

    # res = "{cfg.res_w}x{cfg.res_h}".format(cfg=cfg)
    # There are two kinds of resolution: The one for which doom creates the img output, here 160x120 is the smallest possible
    # The other is the resize resolution which will be taken from the cfg.res_w/h

    return make_doom_env_impl(
        spec, cfg=cfg, env_config=env_config, render_mode=render_mode, **kwargs
    )


def register_retinal_env(
    scene_name: str, cache_dir: str, input_satiety: bool, allow_backwards: bool = True
):
    if not os.path.isabs(cache_dir):
        # make path absolute by making it relative to the path of this file
        # TODO: Discuss whether this is desired behaviour...
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", cache_dir)
    cfg_path = os.path.join(cache_dir, "scenarios", scene_name + ".cfg")

    env_spec = retinal_doomspec(scene_name, cfg_path, input_satiety, allow_backwards)
    make_env_func = functools.partial(make_retinal_env_from_spec, env_spec)
    register_env(env_spec.name, make_env_func)
