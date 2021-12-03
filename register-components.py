"""
From the root of Sample Factory repo this can be run as:
python -m sample_factory_examples.train_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sample_factory_examples.enjoy_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example

"""

import sys

from retina_rl.encoder import *

def main():
    """Script entry point."""
    register_custom_encoder('retina_encoder', RNNEncoder)

if __name__ == '__main__':
    sys.exit(main())
