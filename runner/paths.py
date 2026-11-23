"""Where runs and the files inside a run directory are located.

The locations themselves are defined only in ``config/base/config.yaml``:

* ``hydra.run.dir`` / ``hydra.sweep.dir`` + ``hydra.sweep.subdir``: run directories
* ``hydra.searchpath``: additional config directories (e.g. ``config/user``)
* ``path.*``: files and directories inside a run directory

This module reads them from there, so that code which runs before hydra has
composed a config (e.g. locating a run to resume) uses the same definitions.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_DIR = PROJECT_ROOT / "config" / "base"
BASE_CONFIG_NAME = "config"

# Files hydra itself writes to its output_subdir (names are fixed by hydra)
HYDRA_CONFIG_FILE = "config.yaml"
HYDRA_META_FILE = "hydra.yaml"
HYDRA_OVERRIDES_FILE = "overrides.yaml"

_INTERPOLATION = re.compile(r"\$\{[^}]*\}")


def load_base_config() -> DictConfig:
    """The unresolved base config (resolving `${hydra:...}` needs a running hydra app)."""
    cfg = OmegaConf.load(BASE_CONFIG_DIR / f"{BASE_CONFIG_NAME}.yaml")
    assert isinstance(cfg, DictConfig)
    return cfg


@dataclass(frozen=True)
class RunFiles:
    """Absolute locations of the files of one run directory."""

    run_dir: Path
    config_dir: Path
    run_config: Path
    launch_overrides: Path
    resume_log: Path
    full_config: Path

    @classmethod
    def from_config(cls, path_cfg: DictConfig, run_dir: Path) -> "RunFiles":
        """Build from the ``path`` node of the base config or of a composed config."""
        return cls(
            run_dir=run_dir,
            config_dir=run_dir / path_cfg.config_dir,
            run_config=run_dir / path_cfg.run_config,
            launch_overrides=run_dir / path_cfg.launch_overrides,
            resume_log=run_dir / path_cfg.resume_log,
            full_config=run_dir / path_cfg.full_config,
        )

    @property
    def hydra_config(self) -> Path:
        return self.config_dir / HYDRA_CONFIG_FILE

    @property
    def hydra_meta(self) -> Path:
        return self.config_dir / HYDRA_META_FILE

    @property
    def hydra_overrides(self) -> Path:
        return self.config_dir / HYDRA_OVERRIDES_FILE


def config_roots(base_cfg: DictConfig, cwd: Path) -> list[Path]:
    """The base config dir plus hydra's (file://) searchpath, relative to the launch dir."""
    searchpath = [
        cwd / entry.removeprefix("file://")
        for entry in base_cfg.hydra.searchpath
        if entry.startswith("file://")
    ]
    return [BASE_CONFIG_DIR, *searchpath]


def run_dir_patterns(base_cfg: DictConfig, run_name: str) -> list[str]:
    """Glob patterns matching the single-run and sweep directories of a run."""
    hydra_cfg = base_cfg.hydra
    templates = [
        _raw(hydra_cfg.run, "dir"),
        f"{_raw(hydra_cfg.sweep, 'dir')}/{_raw(hydra_cfg.sweep, 'subdir')}",
    ]
    return [
        _INTERPOLATION.sub("*", t.replace("${run_name}", run_name)) for t in templates
    ]


def find_run_dirs(base_cfg: DictConfig, run_name: str, cwd: Path) -> list[Path]:
    """All existing run directories with this name (relative templates use cwd)."""
    return sorted(
        path
        for pattern in run_dir_patterns(base_cfg, run_name)
        for path in cwd.glob(pattern)
        if path.is_dir()
    )


def find_run_dir(base_cfg: DictConfig, run_name: str, cwd: Path) -> Path | None:
    """The run directory with this name, None if there is none."""
    run_dirs = find_run_dirs(base_cfg, run_name, cwd)
    if len(run_dirs) > 1:
        raise ValueError(
            f"Run name '{run_name}' is ambiguous: {[str(d) for d in run_dirs]}"
        )
    return run_dirs[0] if run_dirs else None


def _raw(node: DictConfig, key: str) -> str:
    """A config value without resolving its interpolations."""
    return str(OmegaConf.to_container(node, resolve=False)[key])
