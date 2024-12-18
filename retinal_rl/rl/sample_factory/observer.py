import multiprocessing
import os

import torch
import wandb
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import debug_log_every_n, log

from hydra.utils import instantiate
from retinal_rl.models.loss import ContextT
from retinal_rl.models.objective import Objective
from retinal_rl.rl.analyze import AnalysesCfg, analyze
from retinal_rl.rl.util import (
    analysis_root,
    plot_path,
    read_analysis_count,
    write_analysis_count,
)
from runner.frameworks.rl.sf_framework import SFFramework

multiprocessing.set_start_method(
    "spawn", force=True
)  # Important.  TODO: Readup on this

### Runner ###


class RetinalAlgoObserver(AlgoObserver):
    """
    AlgoObserver that runs analysis at specified times.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.freq = cfg.analysis_freq
        self.current_process = None
        self.queue = multiprocessing.Queue()

        # get analysis count
        if not os.path.exists(analysis_root(cfg)):
            os.makedirs(analysis_root(cfg))

        acount = read_analysis_count(cfg)

        self.steps_complete = acount

    def analyze(self, queue : multiprocessing.Queue[None]):
        """Run analysis in a separate process."""

        brain = SFFramework.load_brain_from_checkpoint(self.cfg)
        objective: Objective[ContextT] = instantiate(self.cfg.objective, brain=brain)
        cfg = AnalysesCfg()
        histories = DDD
        epoch = AAA
        analyze(cfg, torch.device('cuda'), brain, objective, histories, epoch, copy_checkpoint=False)
        # envstps = analyze(self.cfg, progress_bar=False)
        # queue.put(envstps, block=False)

    def on_training_step(
        self, runner: Runner, _
    ) -> None:  # TODO: deprecated and will be refactored anyway
        """Called after each training step."""
        # TODO: Check and refactor
        if self.current_process is None:
            total_env_steps = sum(runner.env_steps.values())
            current_step = total_env_steps // self.freq

            msg = (
                "RETINAL RL: No analysis running. current_step = %d, steps_complete = %d"
                % (current_step, self.steps_complete)
            )
            debug_log_every_n(100, msg)

            if current_step >= self.steps_complete:
                # run analysis in a separate process
                log.debug(
                    "RETINAL RL: current_step >= self.steps_complete, launching analysis process..."
                )
                self.current_process = multiprocessing.Process(
                    target=self.analyze, args=(self.queue,)
                )
                self.current_process.start()

        else:
            if not self.current_process.is_alive():
                if self.current_process.exitcode == 0:
                    log.debug(
                        "RETINAL RL: Analysis process finished successfully. Retrieving envstps..."
                    )
                    envstps = self.queue.get()
                    ana_name = "env_steps-" + str(envstps)

                    if self.cfg.with_wandb:
                        log.debug("RETINAL RL: Uploading plots to wandb...")

                        pltpth = plot_path(self.cfg, ana_name)
                        # Recursively list all files in pltpth
                        for path, _, files in os.walk(pltpth):
                            # upload all pngs to wandb
                            for f in files:
                                if f.endswith(".png"):
                                    log.debug("RETINAL RL: Uploading %s", f)
                                    wandb.log({f: wandb.Image(os.path.join(path, f))})
                            # Upload video to wandb
                            for f in files:
                                if f.endswith(".mp4"):
                                    log.debug("RETINAL RL: Uploading %s", f)
                                    wandb.log({f: wandb.Video(os.path.join(path, f))})

                    self.steps_complete += 1
                    write_analysis_count(self.cfg, self.steps_complete)

                self.current_process.join()
                self.current_process = None
