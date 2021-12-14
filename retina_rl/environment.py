"""
retina_rl library

"""
import os
from os.path import join

from sample_factory.envs.doom.doom_utils import DoomSpec, register_additional_doom_env
from sample_factory.envs.doom.action_space import doom_action_space_extended
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args

def register_environment(nm):

    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder

    spec = DoomSpec(
            'doom_' + nm,
            join(os.path.abspath('scenarios'), nm + '.cfg'),  # use your custom cfg here
            doom_action_space_extended(),
            reward_scaling=0.01,
            )

    print(join(os.path.abspath('scenarios'), nm + '.cfg'))

    register_additional_doom_env(spec)

def register_environments():

    register_environment('gathering_r25_b25_g250')
    register_environment('apple_gathering_r30_b0_g0')
    register_environment('apple_gathering_r30_b0_100')
    register_environment('apple_gathering_r30_b2_g0')
    register_environment('apple_gathering_r30_b2_g100')
    

def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).
    """
    parser = arg_parser(argv, evaluation=evaluation)

    # Parse args for rvvs model from Lindsey et al 2019
    parser.add_argument('--global_channels', type=int, default=16, help='Standard number of channels in CNN layers')
    parser.add_argument('--retinal_bottleneck', type=int, default=4, help='Number of channels in retinal bottleneck')
    parser.add_argument('--vvs_depth', type=int, default=1, help='Number of CNN layers in the ventral stream network')
    parser.add_argument('--kernel_size', type=int, default=7, help='Size of CNN filters')

    # SampleFactory parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg
