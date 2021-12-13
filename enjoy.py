import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy
from retina_rl.environment import custom_parse_args
from retina_rl.encoders import register_encoders
from retina_rl.environment import custom_parse_args,register_environments


def main():
    """Script entry point."""
    register_encoders()
    register_environments()
    cfg = custom_parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
