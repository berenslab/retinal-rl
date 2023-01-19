import os
from os.path import join
import functools

from sample_factory.envs.env_utils import register_env

from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec
from sf_examples.vizdoom.doom.doom_utils import DoomSpec, make_doom_env_from_spec


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


### Retinal Environments ###


RETINAL_ENVS = [

    DoomSpec(
        "appmnist_apples_gathering_01",
        join(os.getcwd(), "scenarios", "appmnist_apples_gathering_01.cfg"),
        doom_action_space_basic(),
        reward_scaling=0.01,
        ) ,
    DoomSpec(
        "appmnist_mnist_gathering_01",
        join(os.getcwd(), "scenarios", "appmnist_mnist_gathering_01.cfg"),
        doom_action_space_basic(),
        reward_scaling=0.01,
        )
]

def register_retinal_envs():
    for env_spec in RETINAL_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)
