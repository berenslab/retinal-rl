"""
retina_rl library

"""

from sf_examples.vizdoom.doom.doom_params import add_doom_env_args,add_doom_env_eval_args

from sample_factory.utils.utils import str2bool


# Batch launcher

import os
import subprocess
import yaml
import hiyapyco

# Load and merge the YAML files
filenames = ["file1.yaml", "file2.yaml", "file3.yaml"]  # replace with your filenames
configs = [yaml.safe_load(open(filename)) for filename in filenames]
merged_config = hiyapyco.load(configs, method=hiyapyco.METHOD_OVERWRITE)

# Construct the job type variable
wandb_job_type = "_".join(os.path.splitext(filename)[0] for filename in filenames)

# Define the experiment variable
experiment = "-".join(
    f"{param_name}_{{{param_name}}}"
    for param_name, param_info in merged_config["parameters"].items()
    if "values" in param_info and len(param_info["values"]) > 1
)

# Update the merged config
merged_config["parameters"]["wandb_job_type"] = {"value": wandb_job_type}
merged_config["parameters"]["experiment"] = {"value": experiment}

# Loop over variables with multiple values
for param_name, param_info in merged_config["parameters"].items():
    if "values" in param_info and len(param_info["values"]) > 1:
        for value in param_info["values"]:
            # Update the value for the current parameter
            merged_config["parameters"][param_name] = {"value": value}

            # Construct the command line arguments
            args = [
                f"--{name} {info['value']}"
                for name, info in merged_config["parameters"].items()
                if "value" in info
            ]

            # Launch the training script
            command = f"python train.py {' '.join(args)}"
            subprocess.run(command, shell=True)



# Default argument overrides

def retinal_override_defaults(parser):
    """Overrides for the sample factory CLI defaults."""
    parser.set_defaults(
        # This block shouldn't be messed with without extensive testing
        ppo_clip_value=0.2,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,

        # Environment defaults we've settled on
        res_h=120,
        res_w=160,
        decorrelate_envs_on_one_worker=False,

        # Wandb stuff
        with_wandb='True',
        wandb_project="retinal-rl",
        wandb_group="free-range",
        wandb_job_type="test",

        # System specific but we'll still set these defaults

        train_for_env_steps=int(1e10),
        batch_size=2048,
        num_workers=20,
        num_envs_per_worker=8,

        # All of these have been through some testing, and work as good defaults
        exploration_loss='symmetric_kl',
        exploration_loss_coeff=0.001,
        with_vtrace=True,
        normalize_returns=False,
        normalize_input=True,
        recurrence=32,
        rollout=32,
        use_rnn=False,
        rnn_size=32,

        # Evaluation-mode stuff
        eval_env_frameskip=1,  # this is for smoother rendering during evaluation
        fps=35,
        max_num_frames=1000,
        max_num_episodes=100,
    )

def add_retinal_env_args(parser):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).
    """

    # Doom args
    add_doom_env_args(parser)
    parser.add_argument('--base_channels', type=int, default=16, help='Standard number of channels in CNN layers')
    parser.add_argument('--retinal_bottleneck', type=int, default=None, help='Number of channels in retinal bottleneck (None is a default which is interpreted by the selected encoder)')
    parser.add_argument('--retinal_stride', type=int, default=1, help='Stride at the first conv layer (\'BC\'), doesnt apply to \'VVS\'')
    parser.add_argument( "--input_satiety", default=False , type=str2bool, help="Whether to input satiety into the FC network layers.")
    parser.add_argument( "--vision_model", default="retinal" , choices=["retinal", "retinal_stride", "prototypical"]
                        , type=str, help="Type of encoder network architecture.")
    parser.add_argument( "--activation", default="elu" , choices=["elu", "relu", "tanh", "softplus", "identity"]
                        , type=str, help="Type of activation function to use.")
    parser.add_argument('--repeat', type=int, default=1, help="Dummy parameter to indicate which repetition we're at in a wandb sweep")
    parser.add_argument('--analysis_freq', type=int, default=int(19e8), help="How often to run analysis (in frames)")
    parser.add_argument("--online_analysis", default=True,type=str2bool, help="Whether to run online analyses of the model during training")
    parser.add_argument("--dry_run", default=False, type=str2bool, help="Only perform a dry run of the config and network analysis, without training or evaluation")


def add_retinal_env_eval_args(parser):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).
    """

    # Doom args
    add_doom_env_eval_args(parser)

    parser.add_argument("--analysis_name", default=None, help="Name of the analysis directory. If None, sets based on number of training steps.")
    parser.add_argument("--receptive_fields", default=True, type=str2bool, help="Analyze receptive fields of network.")
    parser.add_argument("--simulate", default=True, type=str2bool, help="Runs simulations and analyses")
    parser.add_argument("--plot", default=True, type=str2bool, help="Generate static plots")
    parser.add_argument("--animate", default=True, type=str2bool, help="Animate 'analysis_out.npy'")
    parser.add_argument("--frame_step", type=int, default=0, help="Which frame of the animation to statically plot")
    parser.add_argument("--sta_repeats", type=int, default=200, help="Number of loops in generating STAs")
