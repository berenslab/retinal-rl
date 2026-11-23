"""Tests for resuming runs by run_name (runner/resume.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.append(".")
from runner import resume
from runner.paths import PROJECT_ROOT, RunFiles, load_base_config

BASE_CFG = load_base_config()
TEMPLATE_USER_CONFIG = PROJECT_ROOT / "resources" / "config_templates" / "user"
EXPERIMENT = "classification-small"  # from the templates, always available


@pytest.fixture
def launch_dir(tmp_path: Path) -> Path:
    """A launch dir like the project root (hydra's searchpath config/user is
    relative to it), with the template user configs instead of the local ones."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "base").symlink_to(PROJECT_ROOT / "config" / "base")
    (tmp_path / "config" / "user").symlink_to(TEMPLATE_USER_CONFIG)
    return tmp_path


def _make_run(launch_dir: Path, name: str, legacy: bool = False) -> RunFiles:
    run_dir = launch_dir / "experiments" / "single-run" / EXPERIMENT / name
    files = RunFiles.from_config(BASE_CFG.path, run_dir)
    files.config_dir.mkdir(parents=True)
    cfg = OmegaConf.create({"command": "train", "a": 1, "nested": {"b": 2}})
    overrides = OmegaConf.create([f"+experiment={EXPERIMENT}", "a=1", "command=train"])
    if legacy:  # only the files hydra writes
        OmegaConf.save(cfg, files.hydra_config)
        OmegaConf.save(overrides, files.hydra_overrides)
    else:
        OmegaConf.save(cfg, files.run_config)
        OmegaConf.save(overrides, files.launch_overrides)
    return files


### Stage 1 (before hydra) ###


@pytest.mark.parametrize(
    "argv",
    [
        ["main.py", f"+experiment={EXPERIMENT}", "command=train"],
        ["main.py", "run_name=does_not_exist", f"+experiment={EXPERIMENT}"],
        ["main.py", "-m", "run_name=run_x"],  # multirun is not handled here
    ],
)
def test_fresh_run(launch_dir, argv):
    _make_run(launch_dir, "run_x")
    assert resume.resume_request(argv, BASE_CFG, launch_dir) is None


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_request(launch_dir, legacy):
    files = _make_run(launch_dir, "run_x", legacy=legacy)
    request = resume.resume_request(
        ["main.py", "run_name=run_x", "nested.b=3"], BASE_CFG, launch_dir
    )

    assert request is not None
    assert request.files == files
    assert request.new_overrides == ["nested.b=3"]
    assert request.hydra_argv == [
        "main.py",
        f"+experiment={EXPERIMENT}",  # group choice for hydra to compose
        "run_name='run_x'",
        f"hydra.run.dir='{files.run_dir}'",
    ]  # values (a=1, command=train) come from the snapshot instead
    # runs started before resume support get their files from hydra's
    assert files.run_config.exists()
    assert files.launch_overrides.exists()


def test_resume_rejects_group_change(launch_dir):
    _make_run(launch_dir, "run_x")
    with pytest.raises(ValueError, match="config group"):
        resume.resume_request(
            ["main.py", "run_name=run_x", "brain=other"], BASE_CFG, launch_dir
        )


### End to end through hydra (real project config, no training) ###

_APP = """
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from runner.paths import BASE_CONFIG_DIR, BASE_CONFIG_NAME, load_base_config
from runner.resume import effective_config, resume_request

OmegaConf.register_new_resolver("eval", eval)
request = resume_request(sys.argv, load_base_config(), Path.cwd())
if request is not None:
    sys.argv = request.hydra_argv

@hydra.main(config_path=str(BASE_CONFIG_DIR), config_name=BASE_CONFIG_NAME, version_base=None)
def app(cfg):
    cfg, _ = effective_config(cfg, request)
    OmegaConf.save(cfg, Path(cfg.path.run_dir) / "effective.yaml", resolve=True)
    OmegaConf.save(cfg, Path(cfg.path.run_dir) / cfg.path.full_config, resolve=True)  # as main.py

app()
"""


def _run_app(launch_dir: Path, *overrides: str) -> None:
    app = launch_dir / "app.py"
    app.write_text(_APP)
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    subprocess.run(
        [sys.executable, str(app), *overrides],
        cwd=launch_dir,
        env=env,
        check=True,
        capture_output=True,
    )


def test_end_to_end(launch_dir):
    _run_app(
        launch_dir,
        f"+experiment={EXPERIMENT}",
        "run_name=run_e2e",
        "command=train",
        "logging.checkpoint_step=7",
    )
    run_dir = launch_dir / "experiments/single-run" / EXPERIMENT / "run_e2e"
    files = RunFiles.from_config(BASE_CFG.path, run_dir)
    first = OmegaConf.load(run_dir / "effective.yaml")
    assert first.logging.checkpoint_step == 7
    assert files.run_config.exists()
    assert files.hydra_config.exists()  # hydra's output_subdir == path.config_dir

    # Resume with only the run name: identical config, no changes logged
    _run_app(launch_dir, "run_name=run_e2e")
    assert OmegaConf.load(run_dir / "effective.yaml") == first
    log = OmegaConf.load(files.resume_log)
    assert len(log) == 1 and log[0].changes == {}

    # Resume with a changed value: applied, logged and snapshot archived
    _run_app(launch_dir, "run_name=run_e2e", "logging.max_checkpoints=9")
    second = OmegaConf.load(run_dir / "effective.yaml")
    assert second.logging.max_checkpoints == 9
    assert second.logging.checkpoint_step == 7
    assert second.command == "train"
    log = OmegaConf.load(files.resume_log)
    assert len(log) == 2
    assert log[1].changes == {"logging.max_checkpoints": {"old": 5, "new": 9}}
    assert (
        len(
            list(
                files.config_dir.glob(
                    f"{files.run_config.stem}.*{files.run_config.suffix}"
                )
            )
        )
        == 1
    )

    # The change persists for later resumes
    _run_app(launch_dir, "run_name=run_e2e")
    assert OmegaConf.load(run_dir / "effective.yaml").logging.max_checkpoints == 9


# path block of runs started before path.config_dir etc. existed (e.g. arrhenius Nov 2026)
_OLD_PATH_BLOCK = {
    "run_dir": "${hydra:runtime.output_dir}",
    "data_dir": "data",
    "checkpoint_dir": "${path.data_dir}/checkpoints",
    "plot_dir": "${path.data_dir}/plots",
    "checkpoint_plot_dir": "${path.plot_dir}/checkpoints",
    "wandb_dir": "${path.run_dir}/wandb",
}


def test_end_to_end_old_run(launch_dir):
    """A run started before resume support: only hydra's files, old path block."""
    _run_app(
        launch_dir,
        f"+experiment={EXPERIMENT}",
        "run_name=run_old",
        "command=train",
        "logging.checkpoint_step=7",
    )
    run_dir = launch_dir / "experiments/single-run" / EXPERIMENT / "run_old"
    files = RunFiles.from_config(BASE_CFG.path, run_dir)
    files.run_config.unlink()
    files.launch_overrides.unlink()
    files.full_config.unlink()
    old_cfg = OmegaConf.load(files.hydra_config)
    old_cfg.path = _OLD_PATH_BLOCK
    old_cfg.run_name = "run_${now:%Y-%m-%d-%H-%M-%S-%f}"  # as stored by old runs
    OmegaConf.save(old_cfg, files.hydra_config)

    _run_app(launch_dir, "run_name=run_old", "logging.max_checkpoints=9")

    effective = OmegaConf.load(run_dir / "effective.yaml")
    assert effective.run_name == "run_old"
    assert effective.logging.checkpoint_step == 7
    assert effective.logging.max_checkpoints == 9
    assert effective.path.data_dir == "data"  # the run's own values are kept
    assert Path(effective.path.run_dir) == run_dir
    assert files.full_config.exists()  # new path keys are available
    log = OmegaConf.load(files.resume_log)
    assert log[0].changes == {"logging.max_checkpoints": {"old": 5, "new": 9}}
