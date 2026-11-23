"""Resume an existing run by passing only its ``run_name``.

Usage::

    python main.py run_name=<existing run> [key=value ...]

If a run directory with that name exists, the run's config snapshot
(``path.run_config``) is used instead of composing a new config from the current
yaml files. Additional value overrides are applied on top of the snapshot. Every
resume, including the resulting config changes, is recorded in
``path.resume_log`` and in wandb.

Two stages, both called from ``main.py``:

1. ``resume_request`` (before hydra): locates the run directory and determines
   the command line hydra is started with: the run's original config-group
   choices (hydra needs them to compose) and the existing run directory.
2. ``effective_config`` (inside the hydra task): for fresh runs it stores the
   snapshot, for resumed runs it builds the config from snapshot + new
   overrides and records the resume.
"""

import copy
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.types import Override
from omegaconf import DictConfig, OmegaConf

from runner import config_util as cu
from runner.paths import PROJECT_ROOT, RunFiles, config_roots, find_run_dir
from runner.util import git_hash

logger = logging.getLogger(__name__)

# Keys that are expected to differ between launches / are not "config changes"
_DIFF_IGNORE_PREFIXES = ("command", "run_name", "path.")
# Overrides hydra sets itself (mode) or that are set when resuming (run dir)
_NOT_RECORDED = ("hydra.mode=", "hydra.run.dir=")


@dataclass(frozen=True)
class ResumeRequest:
    run_name: str
    files: RunFiles
    new_overrides: list[str]  # value overrides to apply on top of the snapshot
    hydra_argv: list[str]  # command line to start hydra with


### Stage 1: before hydra ###


def resume_request(
    argv: list[str], base_cfg: DictConfig, cwd: Path
) -> ResumeRequest | None:
    """The request to resume an existing run, None if argv starts a fresh run."""
    flags, override_strs = cu.split_cli_args(argv[1:])
    overrides = cu.parse_overrides(override_strs)
    run_name = cu.override_value(overrides, "run_name")

    run_dir = None
    if run_name is not None and not cu.is_multirun(flags):
        run_dir = find_run_dir(base_cfg, str(run_name), cwd)

    request = None
    if run_dir is not None:
        files = RunFiles.from_config(base_cfg.path, run_dir)
        groups = cu.config_groups(config_roots(base_cfg, cwd))
        _check_no_group_change(overrides, groups, str(run_name))
        _ensure_launch_files(files)
        request = ResumeRequest(
            run_name=str(run_name),
            files=files,
            new_overrides=[
                o.input_line for o in overrides if o.key_or_group != "run_name"
            ],
            hydra_argv=[
                argv[0],
                *flags,
                *_hydra_overrides(files, groups, str(run_name)),
            ],
        )
    return request


def _check_no_group_change(
    overrides: list[Override], groups: set[str], run_name: str
) -> None:
    for o in overrides:
        if cu.is_group_override(o, groups):
            raise ValueError(
                f"Can not change config group '{o.input_line}' when resuming "
                f"run '{run_name}'. Only value overrides are supported."
            )


def _hydra_overrides(files: RunFiles, groups: set[str], run_name: str) -> list[str]:
    original = cu.parse_overrides(_load_overrides(files.launch_overrides))
    kept = [
        o.input_line
        for o in original
        if cu.is_group_override(o, groups) or o.key_or_group.startswith("hydra.")
    ]
    return [
        *kept,
        f"run_name={cu.quote(run_name)}",
        f"hydra.run.dir={cu.quote(str(files.run_dir))}",
    ]


def _ensure_launch_files(files: RunFiles) -> None:
    """take snapshot and launch overrides, before hydra overwrites them."""
    if not files.run_config.exists():
        if not files.hydra_config.exists():
            raise FileNotFoundError(f"No config snapshot in {files.config_dir}")
        shutil.copyfile(files.hydra_config, files.run_config)
        print(f"Created {files.run_config} from {files.hydra_config}")
    if not files.launch_overrides.exists():
        _save_overrides(files.launch_overrides, _legacy_launch_overrides(files))


def _legacy_launch_overrides(files: RunFiles) -> list[str]:
    if files.hydra_meta.exists():  # contains hydra and task overrides
        recorded = OmegaConf.load(files.hydra_meta).hydra.overrides
        overrides = [str(o) for o in [*recorded.hydra, *recorded.task]]
    elif files.hydra_overrides.exists():
        overrides = _load_overrides(files.hydra_overrides)
    else:
        raise FileNotFoundError(f"No launch overrides in {files.config_dir}")
    return overrides


def _load_overrides(path: Path) -> list[str]:
    return [str(o) for o in OmegaConf.load(path)]


def _save_overrides(path: Path, overrides: list[str]) -> None:
    recorded = [o for o in overrides if not o.startswith(_NOT_RECORDED)]
    OmegaConf.save(OmegaConf.create(recorded), path)


### Stage 2: inside the hydra task ###


def effective_config(
    cfg: DictConfig, request: ResumeRequest | None
) -> tuple[DictConfig, dict[str, Any] | None]:
    """The config to run with, and the resume log entry (None for fresh runs)."""
    if request is None:
        _store_launch_files(cfg)
        result: tuple[DictConfig, dict[str, Any] | None] = (cfg, None)
    else:
        result = _resumed_config(request, cfg)
    return result


def _store_launch_files(cfg: DictConfig) -> None:
    files = RunFiles.from_config(cfg.path, Path(cfg.path.run_dir))
    if not files.run_config.exists():
        OmegaConf.save(cfg, files.run_config, resolve=False)
        recorded = HydraConfig.get().overrides
        _save_overrides(files.launch_overrides, [*recorded.hydra, *recorded.task])


def _resumed_config(
    request: ResumeRequest, composed_cfg: DictConfig
) -> tuple[DictConfig, dict[str, Any]]:
    snapshot = OmegaConf.load(request.files.run_config)
    assert isinstance(snapshot, DictConfig)
    _add_missing_path_keys(snapshot, composed_cfg)

    cfg = copy.deepcopy(snapshot)
    cfg.run_name = request.run_name  # legacy snapshots contain ${now:...}
    # launch scripts pass the same +key=v on every (re)start: the snapshot of a
    # resumed run already has the key, so it is set instead of rejected
    cu.apply_overrides(cfg, request.new_overrides, add_existing=True)
    OmegaConf.set_struct(cfg, True)  # behave like a hydra-composed config

    changes = cu.config_diff(snapshot, cfg, _DIFF_IGNORE_PREFIXES)
    entry = _resume_entry(request, changes)
    _append_resume_log(request.files.resume_log, entry)
    _report_changes(changes)
    if changes:
        _update_snapshot(request.files.run_config, cfg)
    return cfg, entry


def _add_missing_path_keys(snapshot: DictConfig, composed_cfg: DictConfig) -> None:
    """Runs started with an older config.yaml lack newer ``path`` entries:
    added unresolved so they refer to the run's actual path values"""
    current = OmegaConf.to_container(composed_cfg.path, resolve=False)
    assert isinstance(current, dict)
    for key in current.keys() - snapshot.path.keys():
        snapshot.path[key] = current[key]


def _resume_entry(
    request: ResumeRequest, changes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
        "git_hash": git_hash(PROJECT_ROOT),
        "overrides": request.new_overrides,
        "changes": changes,
    }


def _append_resume_log(log_file: Path, entry: dict[str, Any]) -> None:
    entries = (
        OmegaConf.to_container(OmegaConf.load(log_file)) if log_file.exists() else []
    )
    assert isinstance(entries, list)
    OmegaConf.save(OmegaConf.create([*entries, entry]), log_file)


def _report_changes(changes: dict[str, dict[str, Any]]) -> None:
    for key, change in changes.items():
        logger.warning(
            f"Resume config change: {key}: {change['old']} -> {change['new']}"
        )
    if any(key.startswith("brain.") for key in changes):
        logger.warning(
            "Brain config changed on resume - loading the checkpoint will fail "
            "if the architecture no longer matches."
        )
    if not changes:
        logger.info("Resuming without config changes.")


def _update_snapshot(snapshot_file: Path, cfg: DictConfig) -> None:
    """Keep the previous snapshot as <name>.<UTC time><suffix> and store the new one."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    archived = snapshot_file.with_name(
        f"{snapshot_file.stem}.{stamp}{snapshot_file.suffix}"
    )
    shutil.copyfile(snapshot_file, archived)
    OmegaConf.save(cfg, snapshot_file, resolve=False)


### wandb ###


def log_resume_to_wandb(entry: dict[str, Any]) -> None:
    """Record a resume and its config changes in the active wandb run."""
    run = wandb.run
    assert run is not None, "wandb has to be initialized"

    # wandb's config.update replaces top-level keys: update the changed leaves
    # in a copy of the (nested) config and send the affected top-level keys.
    config = run.config.as_dict()
    changes: dict[str, dict[str, Any]] = entry["changes"]
    for key, change in changes.items():
        new = None if change["new"] == cu.UNSET else change["new"]
        cu.set_nested(config, key, new)
    changed_top_level = {key.split(".")[0] for key in changes}

    resume_log = [*config.get("resume_log", []), entry]
    run.config.update(
        {**{k: config[k] for k in changed_top_level}, "resume_log": resume_log},
        allow_val_change=True,
    )
    run.summary["resume_count"] = len(resume_log)
    run.tags = tuple(set(run.tags or ()) | {"resumed"})
