"""Tests for runner/config_util.py."""

import sys

import pytest
from omegaconf import OmegaConf

sys.path.append(".")
from runner import config_util as cu


def test_split_cli_args():
    flags, overrides = cu.split_cli_args(
        ["-m", "a=1", "--config-name", "other", "+b=2", "-cp", "dir"]
    )
    assert flags == ["-m", "--config-name", "other", "-cp", "dir"]
    assert overrides == ["a=1", "+b=2"]
    assert cu.is_multirun(flags)


def test_override_value():
    overrides = cu.parse_overrides(["run_name=x", "a=1", "run_name='y'"])
    assert cu.override_value(overrides, "run_name") == "y"
    assert cu.override_value(overrides, "missing") is None


def test_quote_roundtrip():
    value = "dir/with 'quote' and \\backslash"
    (override,) = cu.parse_overrides([f"key={cu.quote(value)}"])
    assert override.value() == value


def test_config_groups(tmp_path):
    for d in ["brain", "hydra/job_logging", "_private", ".hidden"]:
        (tmp_path / d).mkdir(parents=True)
    assert cu.config_groups([tmp_path, tmp_path / "missing"]) == {
        "brain",
        "hydra",
        "hydra/job_logging",
    }


def test_apply_overrides_and_diff():
    old = OmegaConf.create(
        {"a": 1, "nested": {"b": 2, "c": "${a}"}, "gone": 0, "command": "train"}
    )
    new = OmegaConf.create(OmegaConf.to_container(old))
    cu.apply_overrides(
        new, ["a=5", "+nested.d=[1,2]", "++nested.b=3", "~gone", "command=analyze"]
    )
    assert new.nested.c == 5  # interpolation follows the override

    assert cu.config_diff(old, new, ignore_prefixes=("command",)) == {
        "a": {"old": 1, "new": 5},
        "nested.b": {"old": 2, "new": 3},
        "nested.c": {"old": 1, "new": 5},
        "nested.d": {"old": cu.UNSET, "new": [1, 2]},
        "gone": {"old": 0, "new": cu.UNSET},
    }


@pytest.mark.parametrize(
    "override, error",
    [("missing=1", "not in the config"), ("+a=1", "already"), ("~x", "delete")],
)
def test_apply_overrides_errors(override, error):
    cfg = OmegaConf.create({"a": 1})
    with pytest.raises(KeyError, match=error):
        cu.apply_overrides(cfg, [override])


def test_apply_overrides_add_existing():
    cfg = OmegaConf.create({"a": 1})
    cu.apply_overrides(cfg, ["+a=2", "+b=3"], add_existing=True)
    assert cfg == {"a": 2, "b": 3}


def test_flatten_and_set_nested():
    tree = {"a": {"b": 1, "e": {}}, "c": 2}
    assert cu.flatten(tree) == {"a.b": 1, "a.e": {}, "c": 2}
    cu.set_nested(tree, "a.b", 3)
    cu.set_nested(tree, "c.d", 4)  # replaces the non-dict parent
    assert tree == {"a": {"b": 3, "e": {}}, "c": {"d": 4}}
