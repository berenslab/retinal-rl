import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy
from retina_rl.environment import custom_parse_args
from retina_rl.encoders import register_custom_encoders
from retina_rl.environment import custom_parse_args,register_custom_doom_environments


def main():
    """Script entry point."""
    register_custom_encoders()
    register_custom_doom_environments()
    cfg = custom_parse_args(evaluation=True)
    print(cfg)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
