import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args


### Main ###


def main():
    """Script entry point."""
    # Register retinal environments and models.
    register_retinal_envs()
    register_retinal_model()

    # Two-pass building parser and returning cfg : Namespace
    parser, _ = parse_sf_args()
    add_retinal_env_args(parser)
    retinal_override_defaults(parser)
    cfg = parse_full_cfg(parser)

    # Allows reading some config variables from string templates - designed for wandb sweeps.
    cfg.train_dir = cfg.train_dir.format(**vars(cfg))
    cfg.experiment = cfg.experiment.format(**vars(cfg))

    # Run simulation
    status = run_rl(cfg)

    return status

if __name__ == "__main__":
    sys.exit(main())
