"""
retina_rl library

"""

from sample_factory.utils.utils import str2bool


def retinal_override_defaults(parser):
    """RL params specific to retinal envs."""
    parser.set_defaults(
        encoder_custom='lindsey',
        hidden_size=64,
        ppo_clip_value=0.2,  # value used in all experiments in the paper
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,
        fps=35,
        exploration_loss='symmetric_kl',
        num_envs_per_worker=10,
        batch_size=2048,
        exploration_loss_coeff=0.001,
        reward_scale=0.1,
        with_wandb='True',
        wandb_tags=['retinal_rl','appo'],
        wandb_project="retinal_rl"
    )

def add_retinal_env_args(parser):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).
    """
    # Parse args for rvvs model from Lindsey et al 2019
    parser.add_argument('--global_channels', type=int, default=32, help='Standard number of channels in CNN layers')
    parser.add_argument('--retinal_bottleneck', type=int, default=4, help='Number of channels in retinal bottleneck')
    parser.add_argument('--vvs_depth', type=int, default=1, help='Number of CNN layers in the ventral stream network')
    parser.add_argument('--kernel_size', type=int, default=7, help='Size of CNN filters')
    parser.add_argument('--retinal_stride', type=int, default=2, help='Stride at the first conv layer (\'BC\'), doesnt apply to \'VVS\'')
    parser.add_argument( "--activation", default="elu" , choices=["elu", "relu", "tanh", "linear"]
                        , type=str, help="Type of activation function to use.")
    parser.add_argument("--greyscale", default=False, type=str2bool
                        , help="Whether to greyscale the input image.")
    parser.add_argument('--shape_reward', type=str2bool, default=True, help='Turns on reward shaping')

    # for analyze script
    parser.add_argument('--analyze_acts', type=str, default='False', help='Visualize activations via gifs and dimensionality reduction; options: \'environment\', \'mnist\' or \'cifar\'') # specific for analyze.py
    parser.add_argument('--analyze_max_num_frames', type=int, default=1e3, help='Used for visualising \'environment\' activations (leave as defult otherwise), normally 100000 works for a nice embedding, but can take time') # specific for analyze.py
    parser.add_argument('--analyze_ds_name', type=str, default='CIFAR', help='Used for visualizing responses to dataset (can be \'MNIST\' or \'CIFAR\'') # specific for analyze.py