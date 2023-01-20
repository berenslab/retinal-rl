import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl


from retinal_rl.system.encoders import register_retinal_models
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.parameters import retinal_override_defaults,add_retinal_env_args

import wandb

## Sweep setup ##

sweep_configuration = {
        'name': 'distractors',
        'description': "Testing how the presence of distractors effects network performance.",
        'method': 'grid',
        'parameters':
        {
            'network': { 'values': ['linear', 'simple', 'complex'] },
            'env': { 'values': ["appmnist_apples_gathering", "appmnist_apples_gathering"] },
            'repeats': { 'distribution': "int_uniform",  'min': 1, 'max': 5 }
            }
        }

sweep_id = wandb.sweep(sweep=sweep_configuration, project='retinal-rl')

## Argument parser ##


## Main ##

def main():

    """Script entry point."""
    register_retinal_envs()
    register_retinal_models()

    parser,_ = parse_sf_args()
    # parameters specific to Doom envs
    add_retinal_env_args(parser)
    # override Doom default values for algo parameters
    retinal_override_defaults(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)
    status = run_rl(cfg)

    return status

if __name__ == "__main__":
    sys.exit(main())
