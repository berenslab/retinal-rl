import functools
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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


class WarpResizeWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, h: int, w: int, warp_exp: float = 2.0):
        super(WarpResizeWrapper, self).__init__(env)

        self.w = w
        self.h = h
        self.warp_exp = warp_exp

        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = {}
            for key, space in env.observation_space.spaces.items():
                new_spaces[key] = self._calc_new_obs_space(space)
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = self._calc_new_obs_space(env.observation_space)

    def _calc_new_obs_space(self, old_space):
        low, high = old_space.low.flat[0], old_space.high.flat[0]

        assert (
            len(old_space.shape) == 3
        ), "Expected observation space to have shape (h, w, channels)"

        channel_last = len(old_space.shape) < 3 or np.argmin(old_space.shape) == 2
        channels = old_space.shape[-1 if channel_last else 0]
        new_shape = (
            [self.h, self.w, channels] if channel_last else [channels, self.h, self.w]
        )

        return spaces.Box(low, high, shape=new_shape, dtype=old_space.dtype)

    @staticmethod
    def center_warp_image(image, out_shape: tuple[int, int] = (60, 80), exp: float = 2):
        """
        Center-warp the image to a specified output size and scale.
        """
        channel_last = len(image.shape) < 3 or np.argmin(image.shape) == 2
        if channel_last:
            h, w = image.shape[0], image.shape[1]
        else:
            h, w = image.shape[1], image.shape[2]
        center = (h // 2, w // 2)  # (height, width)

        out_shape_half = (out_shape[0] // 2, out_shape[1] // 2)
        row_even = out_shape[0] % 2
        col_even = out_shape[1] % 2
        row_idx = np.round(
            (np.arange(0, out_shape_half[0]) / (out_shape_half[0] - 1)) ** exp
            * (center[0] - (1+row_even))
        ).astype(int)  # Generate indices for rows and columns
        col_idx = np.round(
            (np.arange(0, out_shape_half[1]) / (out_shape_half[1] - 1)) ** exp
            * (center[1] - (1+col_even))
        ).astype(int)

        # ensure difference is at least 1 pixel
        row_inc = np.arange(len(row_idx))
        col_inc = np.arange(len(col_idx))
        row_idx[row_idx < row_inc] = row_inc[row_idx < row_inc]
        col_idx[col_idx < col_inc] = col_inc[col_idx < col_inc]

        h = center[0] - row_idx[::-1] - 1 -row_even
        w = center[1] - col_idx[::-1] - 1 -col_even
        if row_even:
            h = np.hstack([h, center[0]-1])
        if col_even:
            w = np.hstack([w, center[1]-1])
        h = np.hstack([h, row_idx + center[0]])
        w = np.hstack([w, col_idx + center[1]])
        if channel_last:
            out = image[h[:, np.newaxis], w[np.newaxis, :]]
        else:
            out = image[:, h[:, np.newaxis], w[np.newaxis, :]]
        return out

    def _convert_obs(self, obs):
        if obs is None:
            return obs

        obs = self.center_warp_image(obs, out_shape=(self.h, self.w), exp=self.warp_exp)
        return obs

    def _observation(self, obs):
        if isinstance(obs, dict):
            new_obs = {}
            for key, value in obs.items():
                new_obs[key] = self._convert_obs(value)
            return new_obs
        else:
            return self._convert_obs(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(obs), reward, terminated, truncated, info


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


### Retinal Environments ###


def retinal_doomspec(
    scene_name: str,
    cfg_path: str,
    sat_in: bool,
    allow_backwards: bool,
    warp_exp: Optional[float] = None,
    warp_w: int = 80,
    warp_h: int = 60,
):
    ewraps = []

    if sat_in:
        ewraps = [(SatietyInput, {})]
    if warp_exp is not None:
        ewraps.append(
            (WarpResizeWrapper, {"h": warp_h, "w": warp_w, "warp_exp": warp_exp})
        )

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
    scene_name: str,
    cache_dir: str,
    input_satiety: bool,
    allow_backwards: bool = True,
    warp_exp: Optional[float] = None,
    warp_h: int = 60,
    warp_w: int = 80,
):
    if not os.path.isabs(cache_dir):
        # make path absolute by making it relative to the path of this file
        # TODO: Discuss whether this is desired behaviour...
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", cache_dir)
    cfg_path = os.path.join(cache_dir, "scenarios", scene_name + ".cfg")

    env_spec = retinal_doomspec(
        scene_name,
        cfg_path,
        input_satiety,
        allow_backwards,
        warp_exp=warp_exp,
        warp_h=warp_h,
        warp_w=warp_w,
    )
    make_env_func = functools.partial(make_retinal_env_from_spec, env_spec)
    register_env(env_spec.name, make_env_func)
