import sys
import os
from multiprocessing import Process

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.misc import ExperimentStatus

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.utils.utils import log

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args
from retinal_rl.analysis.util import get_analysis_times,analysis_root

from analyze import analyze


### Runner ###


class RetinalAlgoObserver(AlgoObserver):
    """
    AlgoObserver that runs analysis at specified times.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.freq = cfg.analysis_freq
        self.current_process = None

        # get analysis times
        if not os.path.exists(analysis_root(cfg)):
            os.makedirs(analysis_root(cfg))

        self.analysis_times = get_analysis_times(cfg)

        self.last_analysis = max(self.analysis_times,default=-1)
        self.steps_complete = 1 + self.last_analysis // self.freq

    def analyze(self):
        """Run analysis in a separate process."""

        analyze(self.cfg)

        #if ex == 0:
        #    log.debug(f"RETINAL RL: Analysis complete at {total_env_steps} env steps.")
        #else:
        #    log.debug(f"RETINAL RL: Analysis failed at {total_env_steps} env steps.")
        #    sys.exit(1)

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        """Called after each training step."""

        if self.current_process is None:

            total_env_steps = sum(runner.env_steps.values())
            current_step = total_env_steps // self.freq

            log.debug("RETINAL RL: No analysis running; current_step = %s, steps_complete = %s",current_step,self.steps_complete)

            if current_step >= self.steps_complete:
                # run analysis in a separate process
                self.current_process = Process(target=self.analyze)
                self.current_process.start()

        else:
            if not self.current_process.is_alive():
                self.current_process.join()
                if self.current_process.exitcode == 0:
                    self.steps_complete += 1
                self.current_process = None





def run_rl(cfg: Config):
    """Run RL training."""
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
