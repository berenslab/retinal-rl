import os
from os.path import join
import functools

from sample_factory.envs.env_utils import register_env
from sample_factory.algo.runners.runner import AlgoObserver, Runner

from sf_examples.vizdoom.doom.doom_utils import make_doom_env_from_spec
from sf_examples.vizdoom.doom.doom_utils import DoomSpec, make_doom_env_from_spec

import gym
from gym.spaces import Discrete

# import necessary
from retinal_rl.analysis.util import get_analysis_times
from multiprocessing import Process


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

### Retinal AlgoObserver ###

class RetinalAlgoObserver(AlgoObserver):
    """
    AlgoObserver that runs analysis at specified times.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.freq = cfg.analysis_freq
        self.current_process = None
        # get analysis times
        self.analysis_times = get_analysis_times(cfg)
        self.last_analysis = 0
        if self.analysis_times is not []: self.last_analysis = max(self.analysis_times)
        self.analysis_step = self.last_analysis // self.freq

    def analyze(self,total_env_steps):
        from analyze import analyze
        analyze(self.cfg)
        self.current_process = None
        self.analysis_step = total_env_steps // self.freq

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        """Called after each training step."""
        if self.current_process is None:
            total_env_steps = sum(runner.env_steps.values())
            if total_env_steps // self.freq >= self.analysis_step:
                # run analysis in a separate process
                self.current_process = Process(target=self.analyze,args=(total_env_steps,))
                self.current_process.start()

### Retinal Environments ###

def retinal_doomspec(scnr,flnm):
    return DoomSpec( scnr
                    , join(os.getcwd(), "scenarios", flnm)
                    , doom_action_space_basic()
                    , reward_scaling=0.01
                    , extra_wrappers=[]
                    )

RETINAL_ENVS = [

    retinal_doomspec("gathering_cifar", "cifar_gathering_01.cfg"),
    retinal_doomspec("gathering_gabors", "gabor_gathering_02.cfg"),
    retinal_doomspec("gathering_apples", "apple_gathering_02.cfg"),
    retinal_doomspec("gathering_mnist", "mnist_gathering_01.cfg"),
    retinal_doomspec("appmnist_apples", "appmnist_apples_gathering_01.cfg"),
    retinal_doomspec("appmnist_mnist", "appmnist_mnist_gathering_01.cfg")

]

def register_retinal_envs():
    for env_spec in RETINAL_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)
