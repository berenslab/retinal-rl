"""
retina_rl library

"""
import os
from os.path import join

from sample_factory.envs.doom.doom_utils import DoomSpec, register_additional_doom_env
from sample_factory.envs.doom.action_space import doom_action_space_extended
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args

def register_custom_doom_environments():

    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder

    smplbtl = DoomSpec(
            'doom_simple_battle',
            join(os.path.abspath('scenarios'), 'simple_battle.cfg'),  # use your custom cfg here
            doom_action_space_extended(),
            reward_scaling=0.01,
            )

    register_additional_doom_env(smplbtl)

    aplpth = join(os.path.abspath('scenarios'), 'apples_gathering_supreme.cfg')
    print(aplpth)
    aplgthr = DoomSpec(
            'doom_apples_gathering_supreme',
            aplpth,  # use your custom cfg here
            doom_action_space_extended(),
            reward_scaling=0.01,
            )

    register_additional_doom_env(aplgthr)

def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).

    """
    parser = arg_parser(argv, evaluation=evaluation)

    # add custom args here
    parser.add_argument('--my_custom_arg', type=int, default=42, help='Any custom arguments users might define')

    # SampleFactory parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg
