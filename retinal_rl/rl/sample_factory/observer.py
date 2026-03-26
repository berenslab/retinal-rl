import multiprocessing
from pathlib import Path

import torch
from hydra.utils import instantiate
from sample_factory.algo.runners.runner import AlgoObserver, Runner, SummaryWriter
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import log

from retinal_rl.models.objective import Objective
from retinal_rl.rl.analyze import AnalysesCfg, analyze
from retinal_rl.rl.loss import RLContext
from retinal_rl.rl.sample_factory.retinal_stats_handler import retinal_stats_handler
from retinal_rl.rl.sample_factory.util import load_brain_from_checkpoint

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
        self.cur_freq = cfg.analysis_freq_start
        self.end_freq = cfg.analysis_freq_end
        self.current_process = None
        self.queue = multiprocessing.Queue()

        # get analysis count
        # if not os.path.exists(analysis_root(cfg)):
        #     os.makedirs(analysis_root(cfg))

        # acount = read_analysis_count(cfg)

        self.next_analysis_step = 0

    def _analyze(self, env_step: int):
        try:
            brain = load_brain_from_checkpoint(self.cfg, latest=True)
            brain.sensors["vision"] = (
                3,
                160,
                160,
            )  # TODO: Hardcore Hack cause there's a bug in rf estimation
            objective: Objective[RLContext] = instantiate(
                self.cfg.objective, brain=brain
            )
            cfg = AnalysesCfg(
                Path(self.cfg.run_dir),
                Path(self.cfg.plot_dir),
                Path(self.cfg.checkpoint_plot_dir),
                Path(self.cfg.data_dir),
                self.cfg.with_wandb,
            )
            epoch = env_step  # use env_step instead of epoch for logging
            analyze(
                cfg,
                torch.device("cuda"),
                brain,
                objective,
                epoch,
                copy_checkpoint=False,
            )
        except Exception as e:
            log.error(f"Analysis failed: {e}")

    def _detach_analyze(self, queue, env_step: int):
        """Run analysis in a separate process."""
        self._analyze(env_step)

        # envstps = analyze(self.cfg, progress_bar=False)
        # queue.put(envstps, block=False)
        queue.put(0, block=False)

    def on_training_step(
        self, runner: Runner, training_iteration_since_resume: int
    ) -> None:  # TODO: deprecated and will be refactored anyway
        """Called after each training step."""
        total_env_steps = sum(runner.env_steps.values())

        if total_env_steps >= self.next_analysis_step:
            # run analysis in a separate process
            log.debug(
                "RETINAL RL: current_step >= self.steps_complete, launching analysis process..."
            )

            self._analyze(total_env_steps)
            self.next_analysis_step = self.next_analysis_step + min(
                self.cur_freq, self.end_freq
            )
            self.cur_freq = self.cur_freq * 2

    # Use extra summaries for now TODO: unify with classification logging
    def extra_summaries(
        self,
        runner: Runner,
        policy_id: PolicyID,
        summary_writer: SummaryWriter,
        env_steps: int,
    ) -> None:
        # report pickup frequencies
        if retinal_stats_handler.pickups:
            total_pickups = max(
                sum(retinal_stats_handler.pickups.values()), 1
            )  # ensure no division by zero

            for object_value, num_pickups in retinal_stats_handler.pickups.items():
                summary_writer.add_scalar(
                    f"pickups/{object_value}", num_pickups / total_pickups, env_steps
                )
                # Clear the stats after logging
                retinal_stats_handler.pickups[object_value] = 0
