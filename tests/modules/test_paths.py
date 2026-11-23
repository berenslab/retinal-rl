"""Tests for runner/paths.py (locations derived from config/base/config.yaml)."""

import sys

import pytest

sys.path.append(".")
from runner import paths

BASE_CFG = paths.load_base_config()


def test_run_dir_patterns_follow_hydra_templates():
    assert paths.run_dir_patterns(BASE_CFG, "run_x") == [
        "experiments/single-run/*/run_x",
        "experiments/multirun/*/sweep/*/run_x",
    ]


def test_find_run_dir(tmp_path):
    assert paths.find_run_dir(BASE_CFG, "run_x", tmp_path) is None

    single = tmp_path / "experiments/single-run/exp/run_x"
    single.mkdir(parents=True)
    assert paths.find_run_dir(BASE_CFG, "run_x", tmp_path) == single

    (tmp_path / "experiments/multirun/exp/sweep/local/run_x").mkdir(parents=True)
    with pytest.raises(ValueError, match="ambiguous"):
        paths.find_run_dir(BASE_CFG, "run_x", tmp_path)


def test_run_files(tmp_path):
    files = paths.RunFiles.from_config(BASE_CFG.path, tmp_path)
    config_dir = tmp_path / BASE_CFG.path.config_dir
    assert files.config_dir == config_dir
    assert files.run_config.parent == config_dir
    assert files.hydra_config == config_dir / paths.HYDRA_CONFIG_FILE


def test_config_roots(tmp_path):
    assert paths.config_roots(BASE_CFG, tmp_path) == [
        paths.BASE_CONFIG_DIR,
        tmp_path / "config" / "user",
    ]
