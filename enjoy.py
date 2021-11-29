import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy
import retina_rl.retina_rl as rl


def main():
    """Script entry point."""
    rl.register_custom_components()
    cfg = rl.custom_parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
