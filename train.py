import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.parameters import retinal_override_defaults,add_retinal_env_args


### Main ###


def main():
    """Script entry point."""
    # Register retinal environments and models.
    register_retinal_envs()
    register_retinal_model()

    # Two-pass building parser and returning cfg : Namespace
    parser, _ = parse_sf_args()
    retinal_override_defaults(parser)
    add_retinal_env_args(parser)
    cfg = parse_full_cfg(parser)

    # Modify config based on sweep parameters
    if cfg.network == "linear":
        cfg.activation = "linear"
        cfg.global_channels = 16
        cfg.retinal_bottleneck = 1
        cfg.retinal_stride = 2
        cfg.vvs_depth = 0
        cfg.kernel_size = 7
    elif cfg.network == "simple":
        cfg.activation = "elu"
        cfg.global_channels = 16
        cfg.retinal_bottleneck = 1
        cfg.retinal_stride = 2
        cfg.vvs_depth = 0
        cfg.kernel_size = 7
    elif cfg.network == "complex":
        cfg.activation = "elu"
        cfg.global_channels = 16
        cfg.retinal_bottleneck = 4
        cfg.retinal_stride = 2
        cfg.vvs_depth = 1
        cfg.kernel_size = 7

    if cfg.experiment == "auto":
        cfg.experiment = cfg.env + "_" + cfg.network + "_" + str(cfg.repeat)

    # Run simulation
    status = run_rl(cfg)

    return status

if __name__ == "__main__":
    sys.exit(main())
