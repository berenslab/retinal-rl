import sys

from retinal_rl.system.arguments import (
    retinal_override_defaults,
    add_retinal_env_args,
    add_retinal_env_eval_args,
)


from retinal_rl.system.exec import analyze
from retinal_rl.util import (
    fill_in_argv_template,
)

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args


def main():
    """Script entry point."""

    # Register retinal environments and models.

    # Parsing args
    argv = sys.argv[1:]
    # Replace string templates in argv with values from argv.
    argv = fill_in_argv_template(argv)

    # Two-pass building parser and returning cfg : Namespace
    parser, cfg = parse_sf_args(argv, evaluation=True)

    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)

    cfg = parse_full_cfg(parser, argv)

    # Run analysis
    analyze(cfg)


if __name__ == "__main__":
    sys.exit(main())
