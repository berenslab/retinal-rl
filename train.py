import functools
import sys
import os
from os.path import join
import gym

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from sf_examples.vizdoom.doom.doom_params import add_doom_env_args
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
from sf_examples.vizdoom.doom.doom_utils import DoomSpec, make_doom_env_from_spec

from gym.spaces import Discrete

from retinal_rl.encoders import make_lindsey_encoder
from retinal_rl.parameters import retinal_override_defaults,add_retinal_env_args

def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_retinal_models():
    global_model_factory().register_encoder_factory(make_lindsey_encoder)

def register_retinal_env1():
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    scenario_absolute_path = join(os.path.dirname(__file__), "scenarios", "appcifar_apples_gathering_06.cfg")
    spec = DoomSpec(
        "doom_appcifar",
        scenario_absolute_path,  # use your custom cfg here
        doom_action_space_basic(),
        reward_scaling=0.01,
    )

    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)

def register_retinal_env2():
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    scenario_absolute_path = join(os.path.dirname(__file__), "scenarios", "apple_gathering_02.cfg")
    spec = DoomSpec(
        "doom_apple_gathering",
        scenario_absolute_path,  # use your custom cfg here
        doom_action_space_basic(),
        reward_scaling=0.01,
    )

    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)

def register_retinal_env3():
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    scenario_absolute_path = join(os.path.dirname(__file__), "scenarios", "mnist_gathering_01.cfg")
    spec = DoomSpec(
        "doom_mnist",
        scenario_absolute_path,  # use your custom cfg here
        doom_action_space_basic(),
        reward_scaling=0.01,
    )

    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)




def register_vizdoom_components():
    register_vizdoom_envs()
    register_retinal_models()


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

def register_custom_doom_env():
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    scenario_absolute_path = join(os.path.dirname(__file__), "custom_env", "custom_doom_env.cfg")
    spec = DoomSpec(
        "doom_my_custom_env",
        scenario_absolute_path,  # use your custom cfg here
        doom_action_space_basic(),
        reward_scaling=0.01,
    )

    # register the env with Sample Factory
    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)

def main():
    """Script entry point."""
    register_vizdoom_components()

    parser, cfg = parse_sf_args()
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    add_retinal_env_args(parser)
    # override Doom default values for algo parameters
    retinal_override_defaults(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)
    register_retinal_env1()
    register_retinal_env2()
    register_retinal_env3()
    register_custom_doom_env()


    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
