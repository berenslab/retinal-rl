import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl
from sample_factory.utils.wandb_utils import init_wandb

from retinal_rl.system.encoders import register_retinal_models
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.parameters import retinal_override_defaults,add_retinal_env_args


### Main ###


def main():
    """Script entry point."""
    # Register retinal environments and models.
    register_retinal_envs()
    register_retinal_models()

    # Two-pass building parser and returning cfg : Namespace
    parser, _ = parse_sf_args()
    retinal_override_defaults(parser)
    add_retinal_env_args(parser)
    cfg = parse_full_cfg(parser)

    # Initialize wandb
    #if cfg.with_wandb:

    #    init_wandb(cfg)  # should be done after modifying configuration

    # Run
    #cfg.with_wandb = False
    print("REPEAT:")
    print(cfg.repeat)
    status = run_rl(cfg)

    return status

if __name__ == "__main__":
    sys.exit(main())
