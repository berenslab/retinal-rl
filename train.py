import sys
import os
import wandb
import multiprocessing
multiprocessing.set_start_method("spawn",force=True)

from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.misc import ExperimentStatus

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.utils.utils import log

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args
from retinal_rl.analysis.util import get_analysis_times,analysis_root,plot_path

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
        self.queue = multiprocessing.Queue()

        # get analysis times
        if not os.path.exists(analysis_root(cfg)):
            os.makedirs(analysis_root(cfg))

        self.analysis_times = get_analysis_times(cfg)

        self.last_analysis = max(self.analysis_times,default=-1)
        self.steps_complete = 1 + self.last_analysis // self.freq

    def analyze(self,queue):
        """Run analysis in a separate process."""

        envstps = analyze(self.cfg)
        queue.put(envstps,block=False)

    def on_training_step(self, runner: Runner, _) -> None:
        """Called after each training step."""

        if self.current_process is None:

            total_env_steps = sum(runner.env_steps.values())
            current_step = total_env_steps // self.freq

            log.debug("RETINAL RL: No analysis running; current_step = %s, steps_complete = %s",current_step,self.steps_complete)

            if current_step >= self.steps_complete:
                # run analysis in a separate process
                log.debug("RETINAL RL: current_step >= self.steps_complete, launching analysis process...")
                self.current_process = multiprocessing.Process(target=self.analyze,args=(self.queue,))
                self.current_process.start()

        else:
            if not self.current_process.is_alive():

                if self.current_process.exitcode == 0:

                    log.debug("RETINAL RL: Analysis process finished successfully. Retrieving envstps...")
                    envstps = self.queue.get()

                    if self.cfg.with_wandb:
                        log.debug("RETINAL RL: Uploading plots to wandb...")

                        pltpth = plot_path(self.cfg,envstps)
                        # load all pngs in the plot directory and upload them to wandb
                        for f in os.listdir(pltpth):
                            if f.endswith(".png"):
                                log.debug("RETINAL RL: Uploading %s",f)
                                wandb.log({f: wandb.Image(os.path.join(pltpth,f))})
                        # load all mp4 files in the plot directory and upload them to wandb
                        for f in os.listdir(pltpth):
                            if f.endswith(".mp4"):
                                log.debug("RETINAL RL: Uploading %s",f)
                                wandb.log({f: wandb.Video(os.path.join(pltpth,f))})

                    self.steps_complete += 1
                self.current_process.join()
                self.current_process = None





def run_rl(cfg: Config):
    """Run RL training."""
    cfg, runner = make_runner(cfg)
    if not(cfg.no_observe):
        runner.register_observer(RetinalAlgoObserver(cfg))

    # here we can register additional message or summary handlers
    # see sf_examples/dmlab/train_dmlab.py for example

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status

def fill_in_argv_template(argv):
    """Replace string templates in argv with values from argv."""

    # Separate out boolean flag arguments
    bool_flags = [a for a in argv if a.startswith("--") and "=" not in a]
    argv = [a for a in argv if not a.startswith("--") or "=" in a]

    # Convert argv into a dictionary
    argv = [a.split('=') for a in argv]
    # Remove dashes from argv
    cfg = dict([[a[0].replace("--",""),a[1]] for a in argv])
    # Replace cfg string templates
    cfg = {k:v.format(**cfg) for k,v in cfg.items()}
    # Convert cfg back into argv
    argv = [f"--{k}={v}" for k,v in cfg.items()]

    return argv + bool_flags



### Main ###


def main():
    """Script entry point."""
    # Register retinal environments and models.
    register_retinal_envs()
    register_retinal_model()

    argv = sys.argv[1:]
    # Convert argv into a dictionary
    argv = fill_in_argv_template(argv)

    # Two-pass building parser and returning cfg : Namespace
    parser, _ = parse_sf_args(argv,evaluation=True)
    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)

    cfg = parse_full_cfg(parser, argv)

    # Run simulation
    status = run_rl(cfg)

    return status

if __name__ == "__main__":
    sys.exit(main())
