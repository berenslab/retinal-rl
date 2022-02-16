"""
retina_rl library

"""
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args

from sample_factory.utils.utils import str2bool


def add_retinal_env_args(env, parser):
    p = parser

    p.add_argument('--timelimit', default=None, type=float, help='Allows to override default match timelimit in minutes')
    p.add_argument('--res_w', default=128, type=int, help='Game frame width after resize')
    p.add_argument('--res_h', default=72, type=int, help='Game frame height after resize')
    p.add_argument('--wide_aspect_ratio', default=False, type=str2bool, help='If true render wide aspect ratio (slower but gives better FOV to the agent)')

def retinal_override_defaults(env, parser):
    """RL params specific to retinal envs."""
    parser.set_defaults(
        encoder_custom='lindsey',
        hidden_size=512,
        ppo_clip_value=0.2,  # value used in all experiments in the paper
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,
        fps=35,
        exploration_loss='symmetric_kl',
        num_envs_per_worker=20,
        batch_size=4096,
        exploration_loss_coeff=0.001,
    )
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
    parser.add_argument('--rf_ratio', type=int, default=3, help='Ratio between RFs of first (BC) and second (RGC) convolutional layer in Mosaic network')
    parser.add_argument('--analyze_acts', type=str, default='False', help='Visualize activations via gifs and dimensionality reduction; options: \'environment\', \'mnist\' or \'cifar\'') # specific for analyze.py
    parser.add_argument('--analyze_max_num_frames', type=int, default=1e3, help='Used for visualising \'environment\' activations (leave as defult otherwise), normally 1e5 works for a nice embedding') # specific for analyze.py
    parser.add_argument('--shape_reward', type=bool, default=True, help='Turns on reward shaping')

    # SampleFactory parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg
