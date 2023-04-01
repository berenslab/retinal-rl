import sys
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.misc import ExperimentStatus

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs,RetinalAlgoObserver
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args


### Runner ###

def run_rl(cfg: Config):
    cfg, runner = make_runner(cfg)
    runner.register_observer(RetinalAlgoObserver(cfg))

    # here we can register additional message or summary handlers
    # see sf_examples/dmlab/train_dmlab.py for example

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status
### Main ###


def main():
    """Script entry point."""
    # Register retinal environments and models.
    register_retinal_envs()
    register_retinal_model()

    # Two-pass building parser and returning cfg : Namespace
    parser, _ = parse_sf_args(evaluation=True)
    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
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
