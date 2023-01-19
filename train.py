import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl

from retinal_rl.system.encoders import register_retinal_models
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.parameters import retinal_override_defaults,add_retinal_env_args


### Main ###


def main():
    """Script entry point."""
    register_retinal_envs()
    register_retinal_models()

    parser, cfg = parse_sf_args()
    add_retinal_env_args(parser)
    # override Doom default values for algo parameters
    retinal_override_defaults(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)
    status = run_rl(cfg)

    return status

if __name__ == "__main__":
    sys.exit(main())
