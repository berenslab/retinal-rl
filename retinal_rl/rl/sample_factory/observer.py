import contextlib
import multiprocessing
import queue
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from pathlib import Path

import torch
from hydra.utils import instantiate
from sample_factory.algo.runners.runner import AlgoObserver, Runner, SummaryWriter
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import log

from retinal_rl.models.objective import Objective
from retinal_rl.plot.log import FigureLogger, PendingWandb, flush_pending_wandb
from retinal_rl.rl.analyze import AnalysesCfg, analyze
from retinal_rl.rl.loss import RLContext
from retinal_rl.rl.sample_factory.retinal_stats_handler import retinal_stats_handler
from retinal_rl.rl.sample_factory.util import load_brain_from_checkpoint

multiprocessing.set_start_method("spawn", force=True)

# (env_step, entries to log to wandb, error message or None)
AnalysisResult = tuple[int, PendingWandb, str | None]
AnalysisFn = Callable[[Config, int, FigureLogger], None]


def run_analysis(cfg: Config, env_step: int, figure_logger: FigureLogger) -> None:
    """Analyze the latest checkpoint. Runs in the analysis process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain = load_brain_from_checkpoint(cfg, latest=True)
    brain.sensors["vision"] = (
        3,
        160,
        160,
    )  # TODO: Hardcore Hack cause there's a bug in rf estimation
    objective: Objective[RLContext] = instantiate(cfg.objective, brain=brain)
    analyses_cfg = AnalysesCfg(
        Path(cfg.run_dir),
        Path(cfg.plot_dir),
        Path(cfg.checkpoint_plot_dir),
        Path(cfg.data_dir),
        cfg.with_wandb,
    )
    epoch = env_step  # use env_step instead of epoch for logging
    analyze(
        analyses_cfg,
        device,
        brain,
        objective,
        epoch,
        copy_checkpoint=False,
        figure_logger=figure_logger,
    )


def _analysis_worker(
    analysis_fn: AnalysisFn,
    cfg: Config,
    env_step: int,
    staging_dir: str,
    result_queue: "multiprocessing.Queue[AnalysisResult]",
) -> None:
    """Entry point of the analysis process. wandb can only be used by the process owning the run, so figures
    are staged on disk and logged by the observer in the main process."""
    figure_logger = FigureLogger(
        cfg.with_wandb,
        Path(cfg.plot_dir),
        Path(cfg.checkpoint_plot_dir),
        Path(cfg.run_dir),
        wandb_staging_dir=Path(staging_dir),
    )
    error = None
    try:
        analysis_fn(cfg, env_step, figure_logger)
    except Exception as e:
        traceback.print_exc()
        error = str(e)
    # also on failure: whatever was produced before the error still gets logged
    result_queue.put((env_step, figure_logger.pending_wandb, error))


### Runner ###
class RetinalAlgoObserver(AlgoObserver):
    """
    AlgoObserver that runs analysis at specified times.

    The analysis runs in a separate process, so it doesn't block the runner's event loop (which, among others,
    has to process the heartbeats of all other components). At most one analysis runs at a time.
    """

    def __init__(self, cfg: Config, analysis_fn: AnalysisFn = run_analysis):
        self.cfg = cfg
        self.cur_freq = cfg.analysis_freq_start
        self.end_freq = cfg.analysis_freq_end
        # how long to wait for a running analysis when training stops, before killing it
        self.stop_timeout: float = (
            cfg.analysis_stop_timeout if "analysis_stop_timeout" in cfg else 600
        )
        self.analysis_fn = analysis_fn

        self.mp_ctx = multiprocessing.get_context("spawn")
        self.queue: multiprocessing.Queue[AnalysisResult] = self.mp_ctx.Queue()
        self.current_process: BaseProcess | None = None
        # env_step -> staging dir of analyses whose result hasn't arrived yet
        self.staging_dirs: dict[int, str] = {}

        self.next_analysis_step = 0

    def _launch_analysis(self, env_step: int):
        staging_dir = tempfile.mkdtemp(prefix="retinal_analysis_")
        self.staging_dirs[env_step] = staging_dir
        process = self.mp_ctx.Process(
            target=_analysis_worker,
            args=(self.analysis_fn, self.cfg, env_step, staging_dir, self.queue),
            daemon=True,
        )
        process.start()
        self.current_process = process

    def _handle_result(self, result: AnalysisResult):
        env_step, pending_wandb, error = result
        staging_dir = self.staging_dirs.pop(env_step, None)
        if error is not None:
            log.error(f"Analysis failed: {error}")
        else:
            log.debug(f"RETINAL RL: analysis at env step {env_step} finished")
        try:
            flush_pending_wandb(
                pending_wandb,
                Path(self.cfg.run_dir),
                Path(staging_dir) if staging_dir else None,
            )
        except Exception as e:
            log.error(f"Logging analysis results to wandb failed: {e}")

    def _collect_results(self):
        """Non-blocking: handle finished analyses and reap the analysis process."""
        while True:
            try:
                result = self.queue.get_nowait()
            except queue.Empty:
                break
            self._handle_result(result)

        if self.current_process is not None and not self.current_process.is_alive():
            exitcode = self.current_process.exitcode
            self.current_process.join()
            self.current_process = None
            if self.staging_dirs:
                # the result may still be in transit through the queue's pipe
                with contextlib.suppress(queue.Empty):
                    self._handle_result(self.queue.get(timeout=5))
            for env_step, staging_dir in self.staging_dirs.items():
                log.error(
                    f"Analysis failed: analysis process for env step {env_step} died without a result "
                    f"(exit code {exitcode})"
                )
                shutil.rmtree(staging_dir, ignore_errors=True)
            self.staging_dirs.clear()

    def on_training_step(
        self, runner: Runner, training_iteration_since_resume: int
    ) -> None:  # TODO: deprecated and will be refactored anyway
        """Called after each training step."""
        self._collect_results()

        total_env_steps = sum(runner.env_steps.values())

        if total_env_steps >= self.next_analysis_step:
            if self.current_process is not None:
                log.warning(
                    f"RETINAL RL: previous analysis still running, skipping analysis at env step {total_env_steps}"
                )
            else:
                log.debug(
                    "RETINAL RL: current_step >= self.steps_complete, launching analysis process..."
                )
                self._launch_analysis(total_env_steps)
            self.next_analysis_step = self.next_analysis_step + min(
                self.cur_freq, self.end_freq
            )
            self.cur_freq = self.cur_freq * 2

    def on_stop(self, runner: Runner) -> None:
        """Give a running analysis a chance to finish, but don't let it keep the job alive."""
        deadline = time.monotonic() + self.stop_timeout
        if self.current_process is not None:
            log.info(
                f"RETINAL RL: waiting up to {self.stop_timeout:.0f}s for the running analysis to finish"
            )
        while self.current_process is not None and time.monotonic() < deadline:
            self._collect_results()
            time.sleep(0.5)

        if self.current_process is not None:
            log.warning("RETINAL RL: analysis did not finish in time, terminating it")
            self.current_process.terminate()
            self.current_process.join(10)
            if self.current_process.is_alive():
                self.current_process.kill()
                self.current_process.join()
        self._collect_results()

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
            pickups = retinal_stats_handler.pickups

            # Reset Pickups dict
            retinal_stats_handler.pickups = dict()

            total_pickups = max(sum(pickups.values()), 1)  # ensure no division by zero

            for object_value, num_pickups in pickups.items():
                summary_writer.add_scalar(
                    f"pickups/{object_value}", num_pickups / total_pickups, env_steps
                )
                summary_writer.add_scalar(
                    f"pickups/abs_{object_value}", num_pickups, env_steps
                )
