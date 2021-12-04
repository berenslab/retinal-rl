"""
retina_rl library

"""

from sample_factory.algorithms.utils.arguments import arg_parser, parse_args


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
