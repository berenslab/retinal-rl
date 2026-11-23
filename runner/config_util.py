"""Generic helpers for hydra command line overrides and OmegaConf configs."""

import re
from pathlib import Path
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import Override
from omegaconf import DictConfig, OmegaConf

UNSET = "<unset>"  # placeholder in config diffs for keys missing on one side

# hydra CLI flags that consume the next argv element
_FLAGS_WITH_VALUE = {
    "--config-path",
    "-cp",
    "--config-name",
    "-cn",
    "--config-dir",
    "-cd",
    "--experimental-rerun",
}
_MULTIRUN_FLAGS = {"-m", "--multirun"}
_MISSING = object()
_BACKSLASHES_BEFORE_QUOTE = re.compile(r"(\\*)('|$)")


### Command line ###


def split_cli_args(args: list[str]) -> tuple[list[str], list[str]]:
    """Split hydra command line arguments into flags (incl. their values) and overrides."""
    flags: list[str] = []
    overrides: list[str] = []
    it = iter(args)
    for arg in it:
        if arg.startswith("-"):
            flags.append(arg)
            if arg in _FLAGS_WITH_VALUE:
                flags.append(next(it, ""))
        else:
            overrides.append(arg)
    return flags, overrides


def is_multirun(flags: list[str]) -> bool:
    return any(f in _MULTIRUN_FLAGS for f in flags)


def parse_overrides(overrides: list[str]) -> list[Override]:
    return OverridesParser.create().parse_overrides(overrides=overrides)


def override_value(overrides: list[Override], key: str) -> Any | None:
    """Value of the (last) override of `key`, None if not overridden."""
    values = [o.value() for o in overrides if o.key_or_group == key]
    return values[-1] if values else None


def quote(value: str) -> str:
    """Quote a string for use as an override value + deal with hydra backslash only before quotes"""
    escaped = _BACKSLASHES_BEFORE_QUOTE.sub(
        lambda m: m.group(1) * 2 + ("\\'" if m.group(2) else ""), value
    )
    return f"'{escaped}'"


def config_groups(config_roots: list[Path]) -> set[str]:
    """Names of all config groups (sub directories) in the given config dirs."""
    return {
        d.relative_to(root).as_posix()
        for root in config_roots
        if root.is_dir()
        for d in root.rglob("*")
        if d.is_dir() and not d.name.startswith((".", "_"))
    }


def is_group_override(override: Override, groups: set[str]) -> bool:
    return override.key_or_group in groups


### Configs ###


def apply_overrides(
    cfg: DictConfig, overrides: list[str], add_existing: bool = False
) -> None:
    """Apply hydra-style value overrides (key=v, +key=v, ++key=v, ~key) in place.

    add_existing: `+key=v` for a key that already exists sets it (like `++`)
    instead of raising, e.g. when re-applying launch overrides to a snapshot.
    """
    for o in parse_overrides(overrides):
        key = o.key_or_group
        exists = OmegaConf.select(cfg, key, default=_MISSING) is not _MISSING
        if o.is_delete():
            if not exists:
                raise KeyError(f"Can not delete '{key}': not in the config")
            _delete(cfg, key)
        elif o.is_add() and exists and not add_existing:
            raise KeyError(f"Can not add '{key}': already in the config")
        elif not (o.is_add() or o.is_force_add()) and not exists:
            raise KeyError(f"Can not override '{key}': not in the config (use +{key})")
        else:
            OmegaConf.update(cfg, key, o.value(), merge=True, force_add=True)


def config_diff(
    old: DictConfig, new: DictConfig, ignore_prefixes: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    """Changed leaves (resolved) as {dot.key: {"old": ..., "new": ...}}."""
    old_flat = flatten(OmegaConf.to_container(old, resolve=True))
    new_flat = flatten(OmegaConf.to_container(new, resolve=True))
    return {
        key: {"old": old_flat.get(key, UNSET), "new": new_flat.get(key, UNSET)}
        for key in sorted(old_flat.keys() | new_flat.keys())
        if not key.startswith(ignore_prefixes)
        and old_flat.get(key, UNSET) != new_flat.get(key, UNSET)
    }


def flatten(container: Any, prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}. Empty dicts and non-dicts are leaves."""
    if isinstance(container, dict) and container:
        flat: dict[str, Any] = {}
        for key, value in container.items():
            flat.update(flatten(value, f"{prefix}{key}."))
    else:
        flat = {prefix.removesuffix("."): container}
    return flat


def set_nested(tree: dict[str, Any], dot_key: str, value: Any) -> None:
    """tree["a"]["b"] = value for dot_key "a.b", creating (or replacing non-dict) parents."""
    *parents, leaf = dot_key.split(".")
    node = tree
    for parent in parents:
        if not isinstance(node.get(parent), dict):
            node[parent] = {}
        node = node[parent]
    node[leaf] = value


def _delete(cfg: DictConfig, key: str) -> None:
    parent_key, _, leaf = key.rpartition(".")
    parent = OmegaConf.select(cfg, parent_key) if parent_key else cfg
    parent.pop(leaf)
