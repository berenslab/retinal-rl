"""Load activation stores written by ``export_activations.py``.

Only depends on numpy, so this file can be copied to wherever the analysis happens.

    from runner.scripts.load_activations import load_activations

    data = load_activations("EXP_DIR/data/activations/gathering-apples__checkpoint_...")
    data.input                    # (T, C, H, W) uint8, raw agent input for every timestep
    data.activations["retina"]    # (T, ...) activations, aligned with data.input
    data.steps["action"]          # (T, n_action_heads); also action_logits, value, reward,
                                  # terminated, truncated, episode, episode_step, run, frame
    data.info["HEALTH"]           # game variables per timestep (pos_x, pos_y, ... as well)
    data.episodes()               # [Episode(episode, start, stop, complete), ...]
    data.manifest                 # all metadata (runs, seeds, checkpoint, brain config, ...)
    load_config(path)             # the experiment's full config (STORE/config/, needs PyYAML)

Loading is lazy by default: ``data.input`` and ``data.activations[...]`` are LazyArrays
that only decompress the shards you index into (recently used shards are cached, see
``cache_size``). They support ``.shape``, ``.dtype``, ``len()``, and numpy-style indexing
along the frame axis (int, slice, int array, bool mask, plus indices on the other axes);
indexing returns ordinary numpy arrays. ``np.asarray(lazy)`` loads everything.

    data.input[10]                          # one frame
    data.activations["retina"][1000:2000]   # a range of frames
    ep = data.episodes()[3]; data.input[ep.start:ep.stop]
    for start, chunk in data.activations["retina"].chunks(): ...   # stream shard-wise

    # restrict to parts of a big store (steps are always loaded eagerly, they are small)
    load_activations(path, layers=["retina"], input=False, frames=(0, 1000), episodes=[3, 4])
    load_activations(path, episodes=[3], lazy=False)   # plain numpy arrays in memory

    # or stream it shard by shard
    for chunk in iter_shards(path, layers=["rnn"]):
        ...

Command line: ``python runner/scripts/load_activations.py STORE`` prints a summary.
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

MANIFEST = "manifest.json"


class Episode(NamedTuple):
    episode: int  # episode id (unique within the store)
    start: int  # index into the loaded arrays
    stop: int  # exclusive
    complete: bool  # False if the recording stopped mid-episode


@dataclass
class ActivationData:
    input: np.ndarray | LazyArray | None
    activations: dict[str, np.ndarray | LazyArray]
    steps: dict[str, np.ndarray]
    manifest: dict[str, Any]

    def __len__(self) -> int:
        return len(self.steps["frame"])

    @property
    def info(self) -> dict[str, np.ndarray]:
        """Game variables per timestep (HEALTH, USER*, pos_*, ...)."""
        return {
            k[len("info/") :]: v for k, v in self.steps.items() if k.startswith("info/")
        }

    @property
    def runs(self) -> list[dict[str, Any]]:
        return self.manifest["runs"]

    def episodes(self) -> list[Episode]:
        """Contiguous episode segments in the loaded data."""
        ep = self.steps["episode"]
        if len(ep) == 0:
            return []
        bounds = np.flatnonzero(np.diff(ep)) + 1
        starts = np.concatenate([[0], bounds])
        stops = np.concatenate([bounds, [len(ep)]])
        done = self.steps["terminated"] | self.steps["truncated"]
        return [
            Episode(int(ep[a]), int(a), int(b), bool(done[b - 1]))
            for a, b in zip(starts, stops)
        ]


def load_config(path: str | Path, name: str = "full_config.yaml") -> dict[str, Any]:
    """Load one of the experiment configs copied into STORE/config/ (needs PyYAML).
    Available: the experiment's config/*.yaml and sf_config.json (sample factory)."""
    file = Path(path) / "config" / name
    if file.suffix == ".json":
        with open(file) as f:
            return json.load(f)
    import yaml  # lazy: only needed for configs

    with open(file) as f:
        return yaml.safe_load(f)


class _Shard:
    """Read access to one shard: ``.files`` and ``shard[key]``, decoded on access.

    Format 2: zip with one deflated member per array (byte-shuffled if multi-byte) and an
    index.json with shape/dtype/shuffle. Format 1: plain .npz."""

    def __init__(self, file: Path):
        self.npz = None
        if file.suffix == ".npz":
            self.npz = np.load(file)
            self.files = list(self.npz.files)
        else:
            self.zip = zipfile.ZipFile(file)
            self.index = json.loads(self.zip.read("index.json"))
            self.files = list(self.index)

    def __getitem__(self, key: str) -> np.ndarray:
        if self.npz is not None:
            return self.npz[key]
        spec = self.index[key]
        dtype = np.dtype(spec["dtype"])
        raw = np.frombuffer(self.zip.read(key), dtype=np.uint8)
        if spec["shuffle"]:
            raw = raw.reshape(dtype.itemsize, -1).T.copy()
        return raw.view(dtype).reshape(spec["shape"])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        (self.npz if self.npz is not None else self.zip).close()


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(Path(path) / MANIFEST) as f:
        return json.load(f)


def _resolve_layers(
    manifest: dict[str, Any], layers: Iterable[str] | None
) -> list[str]:
    available = list(manifest["layers"])
    if layers is None:
        return available
    resolved = []
    for name in layers:
        matches = [name_ for name_ in available if name_ == name] or [
            name_ for name_ in available if name_.endswith("." + name)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Layer '{name}' {'is ambiguous' if matches else 'not found'}; available: {available}"
            )
        resolved.append(matches[0])
    return resolved


def _frame_range(
    frames: tuple[int | None, int | None] | slice | None, num_frames: int
) -> tuple[int, int]:
    if isinstance(frames, slice):
        if frames.step not in (None, 1):
            raise ValueError("frame slices with a step are not supported")
        frames = (frames.start, frames.stop)
    start, stop = frames if frames is not None else (None, None)
    return (0 if start is None else start, num_frames if stop is None else stop)


class _Segment(NamedTuple):
    """The selected frames of one shard."""

    file: Path
    idx: np.ndarray | slice  # selected local indices within the shard
    steps: dict[str, np.ndarray]


def _as_slice(idx: np.ndarray) -> np.ndarray | slice:
    if len(idx) and idx[-1] - idx[0] == len(idx) - 1:
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _plan(
    path: Path,
    manifest: dict[str, Any],
    frames: tuple[int, int] | slice | None,
    runs: Iterable[int] | None,
    episodes: Iterable[int] | None,
) -> list[_Segment]:
    """Select frames shard by shard; loads the (small) per-step data of selected shards."""
    f_start, f_stop = _frame_range(frames, manifest["num_frames"])
    run_set = None if runs is None else set(runs)
    ep_arr = None if episodes is None else np.asarray(list(episodes))
    if ep_arr is not None and len(ep_arr) == 0:
        return []

    segments = []
    for shard in manifest["shards"]:
        if run_set is not None and shard["run"] not in run_set:
            continue
        a, b = max(f_start, shard["start"]), min(f_stop, shard["stop"])
        if a >= b:
            continue
        if ep_arr is not None and "episodes" in shard:
            first, last = shard["episodes"]
            if not ((ep_arr >= first) & (ep_arr <= last)).any():
                continue
        file = path / shard["file"]
        with _Shard(file) as sh:
            steps = {
                k[len("steps/") :]: sh[k] for k in sh.files if k.startswith("steps/")
            }
        steps["frame"] = np.arange(shard["start"], shard["stop"])
        idx = np.arange(a - shard["start"], b - shard["start"])
        if ep_arr is not None:
            idx = idx[np.isin(steps["episode"][idx], ep_arr)]
            if len(idx) == 0:
                continue
        sel = _as_slice(idx)
        segments.append(_Segment(file, sel, {k: v[sel] for k, v in steps.items()}))
    return segments


class _MemberCache:
    """LRU cache of decoded shard members, bounded in bytes."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.items: OrderedDict[tuple[Path, str], np.ndarray] = OrderedDict()
        self.nbytes = 0

    def get(self, file: Path, key: str) -> np.ndarray:
        if (file, key) in self.items:
            self.items.move_to_end((file, key))
            return self.items[(file, key)]
        with _Shard(file) as sh:
            arr = sh[key]
        arr.flags.writeable = False  # handed out as views; protect the cached copy
        if arr.nbytes <= self.max_bytes:
            self.items[(file, key)] = arr
            self.nbytes += arr.nbytes
            while self.nbytes > self.max_bytes:
                self.nbytes -= self.items.popitem(last=False)[1].nbytes
        return arr


def _frame_indices(first: Any, n: int) -> np.ndarray:
    """Global frame indices selected by a slice / int array / bool mask."""
    if isinstance(first, slice):
        return np.arange(*first.indices(n))
    if first is None:
        raise IndexError("np.newaxis on the frame axis is not supported")
    idx = np.asarray(first)
    if idx.size == 0:
        return np.empty(0, np.int64)
    if idx.dtype == bool:
        if idx.shape != (n,):
            raise IndexError(f"boolean index must have shape ({n},)")
        return np.flatnonzero(idx)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise IndexError("only 1-d integer or boolean arrays index the frame axis")
    idx = np.where(idx < 0, idx + n, idx)
    if len(idx) and (idx.min() < 0 or idx.max() >= n):
        raise IndexError(f"index out of range for {n} frames")
    return idx


class LazyArray:
    """An array of frames spread over shards, decompressed on access.

    Index it like a numpy array; the first axis is the frame axis. Results are numpy arrays
    (single frames may be read-only views into the cache; copy them before modifying)."""

    def __init__(
        self,
        key: str,
        segments: list[_Segment],
        spec: dict[str, Any],
        cache: _MemberCache,
    ):
        self.key = key
        self.segments = segments
        self.cache = cache
        lengths = [len(s.steps["frame"]) for s in segments]
        self.offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
        self.shape = (int(self.offsets[-1]), *spec["shape"])
        self.dtype = np.dtype(spec["dtype"])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (
            f"LazyArray('{self.key.removeprefix('act/')}', shape={self.shape}, dtype={self.dtype}, "
            f"{len(self.segments)} shards, {self.nbytes / 2**30:.2f} GiB)"
        )

    def _segment(self, s: int) -> np.ndarray:
        """Selected frames of segment s."""
        seg = self.segments[s]
        return self.cache.get(seg.file, self.key)[seg.idx]

    def chunks(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (first frame index, array) per shard; memory stays bounded."""
        for s in range(len(self.segments)):
            yield int(self.offsets[s]), self._segment(s)

    def __iter__(self) -> Iterator[np.ndarray]:
        for _, chunk in self.chunks():
            yield from chunk

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = self[:]
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def __getitem__(self, key: Any) -> np.ndarray:
        key = key if isinstance(key, tuple) else (key,)
        if not key or key[0] is Ellipsis:
            first, rest = slice(None), key
        else:
            first, rest = key[0], key[1:]
        n = len(self)

        if isinstance(first, (int, np.integer)):
            i = int(first) + n if first < 0 else int(first)
            if not 0 <= i < n:
                raise IndexError(f"index {first} out of range for {n} frames")
            s = int(np.searchsorted(self.offsets, i, "right")) - 1
            return self._segment(s)[i - self.offsets[s]][rest]

        idx = _frame_indices(first, n)
        rest = (slice(None), *rest)
        seg = np.searchsorted(self.offsets, idx, "right") - 1
        out = None
        for s in np.unique(seg):
            mask = seg == s
            part = self._segment(s)[_as_slice(idx[mask] - self.offsets[s])][rest]
            if out is None:
                out = np.empty((len(idx), *part.shape[1:]), part.dtype)
            out[mask] = part
        if out is None:  # nothing selected
            out = np.empty((0, *self.shape[1:]), self.dtype)[rest]
        return out


def iter_shards(
    path: str | Path,
    layers: Iterable[str] | None = None,
    input: bool = True,
    frames: tuple[int, int] | slice | None = None,
    runs: Iterable[int] | None = None,
    episodes: Iterable[int] | None = None,
) -> Iterator[ActivationData]:
    """Yield the (selected part of the) store shard by shard as numpy arrays.
    Arguments as in load_activations."""
    path = Path(path)
    manifest = load_manifest(path)
    layer_names = _resolve_layers(manifest, layers)
    for seg in _plan(path, manifest, frames, runs, episodes):
        with _Shard(seg.file) as sh:
            yield ActivationData(
                input=sh["input"][seg.idx] if input else None,
                activations={k: sh[f"act/{k}"][seg.idx] for k in layer_names},
                steps=seg.steps,
                manifest=manifest,
            )


def load_activations(
    path: str | Path,
    layers: Iterable[str] | None = None,
    input: bool = True,
    frames: tuple[int, int] | slice | None = None,
    runs: Iterable[int] | None = None,
    episodes: Iterable[int] | None = None,
    lazy: bool = True,
    cache_size: float = 2**30,
) -> ActivationData:
    """Open an activation store.

    Args:
        path: store directory (the one containing manifest.json).
        layers: layers to include (default: all). Unique suffixes work, e.g. 'bipolar_activation'.
        input: whether to include the raw input frames.
        frames: (start, stop) or slice of global frame indices.
        runs: only include frames recorded in these export runs.
        episodes: only include frames from these episode ids.
        lazy: if True (default), input and activations are LazyArrays that decompress shards
            on access; steps are always loaded. If False, everything is loaded into memory.
        cache_size: bytes of decompressed shard data kept in memory when lazy.
    """
    path = Path(path)
    manifest = load_manifest(path)
    layer_names = _resolve_layers(manifest, layers)
    segments = _plan(path, manifest, frames, runs, episodes)

    step_specs = {**manifest["steps"], "frame": {"shape": [], "dtype": "int64"}}
    steps = {
        k: np.concatenate([s.steps[k] for s in segments])
        if segments
        else np.empty((0, *spec["shape"]), spec["dtype"])
        for k, spec in step_specs.items()
    }
    specs = {f"act/{k}": manifest["layers"][k] for k in layer_names}
    if input:
        specs["input"] = manifest["input"]
    cache = _MemberCache(int(cache_size) if lazy else 0)
    arrays: dict[str, Any] = {
        k: LazyArray(k, segments, spec, cache) for k, spec in specs.items()
    }
    if not lazy:
        arrays = {k: v[:] for k, v in arrays.items()}
    return ActivationData(
        input=arrays["input"] if input else None,
        activations={k: arrays[f"act/{k}"] for k in layer_names},
        steps=steps,
        manifest=manifest,
    )


def summary(path: str | Path) -> str:
    path = Path(path)
    m = load_manifest(path)
    size = sum(f.stat().st_size for f in path.glob("shard_*"))
    lines = [
        f"Store:       {path}",
        f"Experiment:  {m['experiment_dir']}",
        "Configs:     "
        + ", ".join(sorted(f.name for f in (path / "config").glob("*"))),
        f"Checkpoint:  {m['model']['checkpoint']} (env_steps {m['model']['env_steps']})",
        f"Frames:      {m['num_frames']} in {m['num_episodes']} episodes, {len(m['shards'])} shards, "
        f"{size / 2**20:.1f} MiB on disk",
        f"Input:       {tuple(m['input']['shape'])} {m['input']['dtype']}",
        "Layers:",
        *[
            f"  {k:<40} {tuple(v['shape'])} {v['dtype']}"
            for k, v in m["layers"].items()
        ],
        "Steps:       " + ", ".join(m["steps"]),
        "Runs:",
        *[
            f"  #{r['run']}: {r['num_frames']} frames from {r['start_frame']}, env={r['env']}, "
            f"seed={r['seed']}, deterministic={r['deterministic']}, status={r.get('status')}, "
            f"{r['started']}"
            for r in m["runs"]
        ],
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} STORE_DIR")
    print(summary(sys.argv[1]))
