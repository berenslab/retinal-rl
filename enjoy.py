import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy

from retinal_rl.environment import register_retinal_environment
from retinal_rl.parameters import custom_parse_args
from retinal_rl.encoders import register_encoders


def main():
    """Script entry point."""
    register_encoders()
    register_retinal_environment()
    cfg = custom_parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
