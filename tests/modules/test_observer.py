import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from sample_factory.utils.attr_dict import AttrDict

from retinal_rl.plot.log import FigureLogger
from retinal_rl.rl.sample_factory.observer import RetinalAlgoObserver

# The analysis functions below run in a spawned process, so they have to be importable module-level functions.


def _log_one_figure(figure_logger: FigureLogger, env_step: int):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figure_logger.log_figure(fig, "receptive_fields", "layer_0", env_step, False)


def slow_analysis(cfg, env_step, figure_logger):
    time.sleep(cfg.sleep)
    _log_one_figure(figure_logger, env_step)


def failing_analysis(_cfg, env_step, figure_logger):
    _log_one_figure(figure_logger, env_step)
    raise ValueError("boom")


def hanging_analysis(_cfg, _env_step, _figure_logger):
    time.sleep(1000)


def crashing_analysis(_cfg, _env_step, _figure_logger):
    os._exit(3)


@pytest.fixture
def cfg(tmp_path: Path) -> AttrDict:
    return AttrDict(
        analysis_freq_start=100,
        analysis_freq_end=1000,
        with_wandb=True,
        run_dir=str(tmp_path),
        plot_dir=str(tmp_path / "plots"),
        checkpoint_plot_dir=str(tmp_path / "checkpoint_plots"),
        sleep=3,
    )


def _runner(env_steps: int):
    return SimpleNamespace(env_steps={0: env_steps})


def _wait_for_analysis(observer: RetinalAlgoObserver, timeout: float = 60):
    deadline = time.monotonic() + timeout
    while observer.current_process is not None and time.monotonic() < deadline:
        observer._collect_results()
        time.sleep(0.1)
    assert observer.current_process is None, "analysis did not finish in time"


@mock.patch("retinal_rl.plot.log.wandb")
def test_analysis_does_not_block_and_results_are_logged_in_main_process(
    wandb_mock, cfg
):
    observer = RetinalAlgoObserver(cfg, analysis_fn=slow_analysis)

    start = time.monotonic()
    observer.on_training_step(_runner(0), 0)
    assert time.monotonic() - start < 1.0
    assert observer.current_process is not None
    staging_dir = Path(observer.staging_dirs[0])

    _wait_for_analysis(observer)

    wandb_mock.log.assert_called_once()
    (logged,), kwargs = wandb_mock.log.call_args
    assert list(logged) == ["Receptive Fields/Layer 0"]
    assert kwargs == {"commit": False}
    png_path = wandb_mock.Image.call_args.args[0]
    assert png_path.startswith(str(staging_dir))
    # staging dir is cleaned up after logging
    assert not staging_dir.exists()
    assert observer.staging_dirs == {}


@mock.patch("retinal_rl.plot.log.wandb")
def test_no_overlapping_analyses(wandb_mock, cfg):
    observer = RetinalAlgoObserver(cfg, analysis_fn=slow_analysis)

    observer.on_training_step(_runner(0), 0)
    first_process = observer.current_process
    # next analysis is due, but the first one is still running: skip it, but advance the schedule
    observer.on_training_step(_runner(150), 1)
    assert observer.current_process is first_process
    assert observer.next_analysis_step == 300

    _wait_for_analysis(observer)
    assert wandb_mock.log.call_count == 1


@mock.patch("retinal_rl.plot.log.wandb")
def test_failing_analysis_is_reported_and_partial_results_logged(wandb_mock, cfg):
    observer = RetinalAlgoObserver(cfg, analysis_fn=failing_analysis)

    with mock.patch("retinal_rl.rl.sample_factory.observer.log") as log_mock:
        observer.on_training_step(_runner(0), 0)
        _wait_for_analysis(observer)

    log_mock.error.assert_called_once_with("Analysis failed: boom")
    wandb_mock.log.assert_called_once()


def test_crashed_analysis_process_is_reaped(cfg):
    observer = RetinalAlgoObserver(cfg, analysis_fn=crashing_analysis)

    with mock.patch("retinal_rl.rl.sample_factory.observer.log") as log_mock:
        observer.on_training_step(_runner(0), 0)
        staging_dir = Path(observer.staging_dirs[0])
        _wait_for_analysis(observer)

    assert "exit code 3" in log_mock.error.call_args.args[0]
    assert not staging_dir.exists()


@mock.patch("retinal_rl.plot.log.wandb")
def test_on_stop_waits_for_running_analysis(wandb_mock, cfg):
    observer = RetinalAlgoObserver(cfg, analysis_fn=slow_analysis)

    observer.on_training_step(_runner(0), 0)
    observer.on_stop(_runner(0))

    assert observer.current_process is None
    wandb_mock.log.assert_called_once()


def test_on_stop_terminates_hanging_analysis(cfg):
    cfg.analysis_stop_timeout = 2
    observer = RetinalAlgoObserver(cfg, analysis_fn=hanging_analysis)

    observer.on_training_step(_runner(0), 0)
    process = observer.current_process
    staging_dir = Path(observer.staging_dirs[0])

    start = time.monotonic()
    observer.on_stop(_runner(0))

    assert time.monotonic() - start < 30
    assert observer.current_process is None
    assert not process.is_alive()
    assert not staging_dir.exists()


@mock.patch("retinal_rl.plot.log.wandb")
def test_figure_logger_staging_mode_does_not_touch_wandb(wandb_mock, tmp_path):
    figure_logger = FigureLogger(
        True, tmp_path, tmp_path, tmp_path, wandb_staging_dir=tmp_path
    )
    _log_one_figure(figure_logger, 0)

    (kind, title, png_path) = figure_logger.pending_wandb[0]
    assert kind == "image"
    assert title == "Receptive Fields/Layer 0"
    assert Path(png_path).is_file()
    wandb_mock.log.assert_not_called()
    wandb_mock.Image.assert_not_called()
