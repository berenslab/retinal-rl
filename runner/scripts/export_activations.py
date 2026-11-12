"""Export network inputs and layer activations of a trained RL agent.

Runs a trained agent (loaded from an experiment directory) in an environment and records, for every
timestep, the raw visual input, the activations of the requested layers, and per-step metadata
(actions, policy logits, value, reward, episode boundaries, game variables like health and position).

Results go to a *store*, a directory ``OUTPUT_DIR/<name>`` (default name:
``<experiment run>__<env>__<checkpoint>``) holding compressed shards, a ``manifest.json``
and ``config/`` (a copy of the experiment's configs). Running the script again with the same output
dir and name appends new frames (new episodes, fresh random seed) to the existing store.

Shards end at episode boundaries (unless an episode exceeds --max-shard-size), so an episode can be
read from a single file. They are compressed and written in a background thread while the
simulation continues, atomically, with the manifest updated after each shard: an interrupted run
(Ctrl+C, crash) keeps everything written up to the last shard. Arrays are byte-shuffled before
deflate compression, which compresses float data noticeably better.

Load results with ``runner/scripts/load_activations.py`` (depends only on numpy)::

    from runner.scripts.load_activations import load_activations
    data = load_activations("path/to/store")

Usage examples::

    # list layer names available for export
    python -m runner.scripts.export_activations EXP_DIR --list-layers

    # 5000 frames of all circuit outputs, on the experiment's own environment
    python -m runner.scripts.export_activations EXP_DIR --num-frames 5000

    # specific layers (circuit outputs and/or sub-modules), different env, float16 storage
    python -m runner.scripts.export_activations EXP_DIR --env gathering-mnist \\
        --layers vision retina retina.bipolar_activation rnn --dtype float16

Timestep alignment (index t)::

    input[t]           raw observation seen by the agent at t (uint8, C x H x W, before transforms)
    activations[l][t]  activation of layer l when processing input[t]
    action[t], action_logits[t], value[t]   outputs for input[t]
    reward[t], terminated[t], truncated[t]  result of executing action[t]
    info/<VAR>[t]      game variables of the state that produced input[t]
    episode[t], episode_step[t], run[t]     bookkeeping
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

FORMAT_VERSION = 2
MANIFEST = "manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


### Arguments ###


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("experiment_dir", type=Path, help="Run directory of the experiment.")
    p.add_argument(
        "--env",
        default=None,
        help="Environment / scenario name to run in (default: the experiment's env).",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layers to record. Brain nodes (circuits or sensors, e.g. 'retina', 'vision', "
        "'rnn_state') record the node output; '<circuit>.<module>' records a sub-module output "
        "(e.g. 'retina.bipolar_activation'; unique suffixes of the module path are accepted). "
        "Default: the circuits on the policy pathway (those feeding into actor/critic, e.g. "
        "not a reconstruction decoder); 'all' records all circuits. See --list-layers.",
    )
    p.add_argument(
        "--list-layers",
        action="store_true",
        help="Print all recordable layers with their shapes and exit.",
    )
    p.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=1000,
        help="Number of timesteps to record in this run (default: 1000).",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Additionally stop once this many episodes have been completed in this run.",
    )
    p.add_argument(
        "--finish-episode",
        action="store_true",
        help="After reaching --num-frames, continue until the current episode ends.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory in which the store directory for this export is created "
        "(default: EXP_DIR/data/activations).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Name of the store directory inside --output-dir "
        "(default: <experiment run dir name>__<env>__<checkpoint>). Existing stores are extended.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing store at the output location instead of extending it.",
    )
    p.add_argument(
        "--checkpoint",
        default="latest",
        help="'latest' (default), 'best', or a path to a .pth checkpoint file.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Take the argmax action instead of sampling from the policy.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for env and action sampling (default: random, recorded in the manifest). "
        "Don't reuse a seed when extending a store unless you want duplicate episodes.",
    )
    p.add_argument(
        "--frameskip",
        type=int,
        default=None,
        help="Env frames per agent step (default: the training value).",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float16",
        help="Storage dtype of activations (default: float16, ~3 significant digits; less than "
        "half the size of float32, which barely compresses).",
    )
    p.add_argument(
        "--compression-level",
        type=int,
        default=6,
        choices=range(0, 10),
        metavar="0-9",
        help="Deflate compression level (default: 6).",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="A shard is closed at the first episode end after it holds this many frames "
        "(default: 1000), so episodes are not split across shards.",
    )
    p.add_argument(
        "--max-shard-size",
        type=int,
        default=5000,
        help="Hard limit of frames per shard (default: 5000); longer episodes are split. "
        "Bounds memory use and what is lost on a crash.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda' or 'cpu' (default: cuda if available).",
    )
    p.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Allow appending to a store that was created with a different checkpoint.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "cache",
        help="Cache directory containing scenarios/ (default: <repo>/cache).",
    )
    return p.parse_args(argv)


### Model and environment ###


def load_experiment_cfg(exp_dir: Path) -> DictConfig:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    for name in ("full_config.yaml", "config.yaml"):
        path = exp_dir / "config" / name
        if path.exists():
            cfg = OmegaConf.load(path)
            break
    else:
        raise FileNotFoundError(
            f"No config/full_config.yaml or config/config.yaml in {exp_dir}"
        )
    if cfg.get("framework") != "rl":
        raise ValueError(
            f"Only RL experiments are supported (framework={cfg.get('framework')})."
        )
    # paths in the saved config are those of the machine the run was trained on
    cfg.path.run_dir = str(exp_dir)
    cfg.path.wandb_dir = str(exp_dir / "wandb")
    cfg.logging.use_wandb = False
    return cfg


def build_sf_cfg(cfg: DictConfig, args: argparse.Namespace, device: str):
    # imported lazily: these pull in sample_factory / vizdoom, which is slow
    from sample_factory.cfg.arguments import load_from_checkpoint

    from retinal_rl.rl.sample_factory.environment import register_retinal_env
    from runner.frameworks.rl.sf_framework import SFFramework

    cfg.system.device = device
    sf_cfg = SFFramework.to_sf_cfg(cfg)
    set_arg = SFFramework._set_cfg_cli_argument
    set_arg(sf_cfg, "with_wandb", False)
    set_arg(sf_cfg, "num_envs_per_worker", 1)
    if args.frameskip is not None:
        set_arg(sf_cfg, "env_frameskip", args.frameskip)

    scenario = Path(args.cache_dir) / "scenarios" / f"{sf_cfg.env}.cfg"
    if not scenario.exists():
        raise FileNotFoundError(
            f"Scenario {scenario} not found. Build it first with doom_creator "
            f"(see doom_creator/compile_scenario.py)."
        )
    register_retinal_env(
        sf_cfg.env,
        str(Path(args.cache_dir).resolve()),
        sf_cfg.input_satiety,
        sf_cfg.allow_backwards,
    )
    return load_from_checkpoint(sf_cfg)


def resolve_checkpoint(sf_cfg, spec: str) -> Path:
    if spec not in ("latest", "best"):
        path = Path(spec).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {path} does not exist.")
        return path
    ckpt_dir = (
        Path(sf_cfg.train_dir)
        / sf_cfg.experiment
        / f"checkpoint_p{sf_cfg.policy_index}"
    )
    pattern = "checkpoint_*.pth" if spec == "latest" else "best_*.pth"
    # names contain zero-padded policy versions, so lexical order is chronological
    candidates = sorted(p for p in ckpt_dir.glob(pattern) if "temp" not in p.name)
    if not candidates:
        raise FileNotFoundError(f"No '{pattern}' checkpoints in {ckpt_dir}.")
    return candidates[-1]


def make_agent(sf_cfg, ckpt_path: Path, device: torch.device):
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.context import global_model_factory
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.utils.attr_dict import AttrDict

    from retinal_rl.rl.sample_factory.models import SampleFactoryBrain

    global_model_factory().register_actor_critic_factory(SampleFactoryBrain)
    env = make_env_func_batched(
        sf_cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    actor_critic = create_actor_critic(sf_cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    ckpt = Learner.load_checkpoint([str(ckpt_path)], device)
    actor_critic.load_state_dict(ckpt["model"])
    return actor_critic, env, int(ckpt.get("env_steps", -1))


### Layer recording ###


@dataclass
class LayerSpec:
    name: str  # name under which it is stored
    node: str | None = None  # brain node whose (first) output is recorded
    circuit: str | None = None  # or: circuit + sub-module path
    module_path: str | None = None


def _drop_batch(t: torch.Tensor) -> torch.Tensor:
    """Batch size is always 1. Note: for modules with sequence-first outputs (nn.GRU) the
    leading dim is the sequence dim, also of size 1 here."""
    return t[0] if t.dim() > 0 and t.shape[0] == 1 else t


def resolve_layers(
    brain, specs: list[str] | None, policy_outputs: tuple[str, ...]
) -> list[LayerSpec]:
    nodes = list(nx.topological_sort(brain.connectome))
    circuits = [n for n in nodes if n in brain.circuits]
    if specs == ["all"]:
        return [LayerSpec(n, node=n) for n in circuits]
    if specs is None:
        keep = set(policy_outputs)
        for out in policy_outputs:
            keep |= nx.ancestors(brain.connectome, out)
        skipped = [n for n in circuits if n not in keep]
        if skipped:
            print(f"Not on the policy pathway, not recorded by default: {skipped}")
        return [LayerSpec(n, node=n) for n in circuits if n in keep]
    out: list[LayerSpec] = []
    for spec in specs:
        if spec in nodes:
            out.append(LayerSpec(spec, node=spec))
            continue
        # '<circuit>.<module path or suffix>', or just a module (path suffix) unique across circuits
        modules = [
            (c, n)
            for c, circ in brain.circuits.items()
            for n, _ in circ.named_modules()
            if n
        ]
        head, _, rest = spec.partition(".")

        def is_suffix(path: str, suffix: str) -> bool:
            return path == suffix or path.endswith("." + suffix)

        full = [(c, n) for c, n in modules if f"{c}.{n}" == spec]
        matches = full or [
            (c, n)
            for c, n in modules
            if (c == head and rest and is_suffix(n, rest)) or is_suffix(n, spec)
        ]
        if len(matches) != 1:
            hint = (
                f"is ambiguous: {[f'{c}.{n}' for c, n in matches]}"
                if matches
                else "not found"
            )
            raise ValueError(
                f"Layer '{spec}' {hint}. Use --list-layers to see options."
            )
        circuit, path = matches[0]
        out.append(LayerSpec(f"{circuit}.{path}", circuit=circuit, module_path=path))
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate layers requested: {names}")
    return out


class ActivationRecorder:
    """Captures activations of the requested layers during a forward pass via hooks."""

    def __init__(self, brain, layers: list[LayerSpec]):
        self.layers = layers
        self.current: dict[str, np.ndarray] = {}
        self.handles = [brain.register_forward_hook(self._brain_hook)]
        for spec in layers:
            if spec.module_path is not None:
                module = brain.circuits[spec.circuit].get_submodule(spec.module_path)
                self.handles.append(
                    module.register_forward_hook(self._module_hook(spec.name))
                )

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        # copy: later in-place ops (e.g. inplace activations) must not alter what we recorded
        return _drop_batch(t.detach()).to("cpu", copy=True).numpy()

    def _brain_hook(self, _module, _inputs, responses):
        for spec in self.layers:
            if spec.node is not None:
                self.current[spec.name] = self._to_numpy(responses[spec.node][0])

    def _module_hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.current[name] = self._to_numpy(output)

        return hook

    def pop(self) -> dict[str, np.ndarray]:
        missing = [s.name for s in self.layers if s.name not in self.current]
        if missing:
            raise RuntimeError(
                f"Layers {missing} were not computed in the forward pass."
            )
        current, self.current = self.current, {}
        return current

    def remove(self):
        for h in self.handles:
            h.remove()


def list_layers(actor_critic, obs, rnn_states):
    """Print all brain nodes and circuit sub-modules with their output shapes."""
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

    brain = actor_critic.brain
    shapes: dict[str, tuple] = {}
    handles = []
    for cname, circuit in brain.circuits.items():
        for mname, module in circuit.named_modules():
            if not mname:
                continue

            def hook(_m, _i, out, key=f"{cname}.{mname}"):
                out = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(out, torch.Tensor):
                    shapes[key] = tuple(_drop_batch(out).shape)

            handles.append(module.register_forward_hook(hook))
    responses = {}
    handles.append(brain.register_forward_hook(lambda _m, _i, r: responses.update(r)))
    with torch.no_grad():
        actor_critic(prepare_and_normalize_obs(actor_critic, obs), rnn_states)
    for h in handles:
        h.remove()

    print("\nBrain nodes (record with the node name; default = all circuits):")
    for node in nx.topological_sort(brain.connectome):
        kind = "circuit" if node in brain.circuits else "sensor"
        print(f"  {node:<40} {kind:<8} {tuple(_drop_batch(responses[node][0]).shape)}")
    print("\nCircuit sub-modules (record with '<circuit>.<module>'):")
    for key, shape in shapes.items():
        print(f"  {key:<48} {shape}")


### Store ###


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _write_json_atomic(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _write_shard_atomic(path: Path, arrays: dict[str, np.ndarray], level: int):
    """Zip archive with one deflated member per array plus index.json (shape, dtype, shuffle).

    Multi-byte arrays are byte-shuffled (all first bytes, then all second bytes, ...): the
    exponent/sign bytes of floats are very regular, which deflate exploits much better when they
    are adjacent. See load_activations.py for the reader."""
    tmp = path.with_suffix(".tmp")
    index = {}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        for key, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            shuffle = arr.dtype.itemsize > 1
            data = (
                arr.view(np.uint8).reshape(-1, arr.dtype.itemsize).T if shuffle else arr
            )
            with zf.open(key, "w", force_zip64=True) as f:
                f.write(np.ascontiguousarray(data).data)
            index[key] = {
                "shape": list(arr.shape),
                "dtype": arr.dtype.str,
                "shuffle": shuffle,
            }
        zf.writestr("index.json", json.dumps(index))
    os.replace(tmp, path)


class Store:
    """Directory of compressed shards + manifest.json; append-only."""

    def __init__(self, path: Path, overwrite: bool):
        self.path = path
        manifest_path = path / MANIFEST
        if overwrite and path.exists():
            if not manifest_path.exists():
                raise ValueError(
                    f"Refusing to delete {path}: it is not an activation store."
                )
            shutil.rmtree(path)
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            if self.manifest.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"{path} uses store format {self.manifest.get('format_version')}, which can "
                    f"still be loaded but not extended (current format: {FORMAT_VERSION}). "
                    "Use another --name or --overwrite."
                )
            self.is_new = False
        elif path.exists() and any(path.iterdir()):
            raise ValueError(
                f"{path} exists, is not empty and is not an activation store."
            )
        else:
            self.manifest = {}
            self.is_new = True
        self.run: dict[str, Any] | None = None

    @property
    def num_frames(self) -> int:
        return self.manifest.get("num_frames", 0)

    @property
    def next_episode(self) -> int:
        return self.manifest.get("num_episodes", 0)

    def init_schema(
        self, schema: dict[str, Any], model: dict[str, Any], allow_model_mismatch: bool
    ):
        if self.is_new:
            self.manifest = {
                "format_version": FORMAT_VERSION,
                "created": _now(),
                **schema,
                "model": model,
                "num_frames": 0,
                "num_episodes": 0,
                "runs": [],
                "shards": [],
            }
            return
        for key in ("input", "layers", "steps"):
            if self.manifest[key] != json.loads(
                json.dumps(schema[key], default=_jsonable)
            ):
                raise ValueError(
                    f"Cannot extend store {self.path}: '{key}' differs.\n"
                    f"  store:    {self.manifest[key]}\n  this run: {schema[key]}\n"
                    "Use the same --layers/--dtype, a different --name/--output-dir, or --overwrite."
                )
        old = self.manifest["model"]
        if (old["checkpoint"], old["env_steps"]) != (
            model["checkpoint"],
            model["env_steps"],
        ):
            msg = (
                f"Store {self.path} was created with checkpoint {old['checkpoint']} "
                f"(env_steps {old['env_steps']}), this run uses {model['checkpoint']} "
                f"(env_steps {model['env_steps']})."
            )
            if not allow_model_mismatch:
                raise ValueError(msg + " Pass --allow-model-mismatch to append anyway.")
            print("WARNING:", msg)

    def start_run(self, info: dict[str, Any]):
        self.path.mkdir(parents=True, exist_ok=True)
        self.run = {
            "run": len(self.manifest["runs"]),
            "start_frame": self.num_frames,
            "num_frames": 0,
            "first_episode": self.next_episode,
            "started": _now(),
            "finished": None,
            **info,
        }
        self.manifest["runs"].append(self.run)
        self._save_manifest()
        return self.run["run"]

    def write_shard(
        self, arrays: dict[str, np.ndarray], num_episodes_total: int, level: int
    ):
        n = len(arrays["steps/episode"])
        if n == 0:
            return
        start = self.num_frames
        fname = f"shard_{start:010d}.zip"
        _write_shard_atomic(self.path / fname, arrays, level)
        episodes = arrays["steps/episode"]
        self.manifest["shards"].append(
            {
                "file": fname,
                "start": start,
                "stop": start + n,
                "run": self.run["run"],
                "episodes": [int(episodes[0]), int(episodes[-1])],
            }
        )
        self.manifest["num_frames"] = start + n
        self.manifest["num_episodes"] = num_episodes_total
        self.run["num_frames"] += n
        self._save_manifest()

    def finish_run(self, status: str, **extra):
        self.run["finished"] = _now()
        self.run["status"] = status
        self.run.update(extra)
        self._save_manifest()

    def _save_manifest(self):
        _write_json_atomic(self.path / MANIFEST, self.manifest)


def copy_configs(exp_dir: Path, sf_cfg, dest: Path):
    """Copy the experiment's hydra configs and the sample factory config into the store.
    Existing copies are kept (the checkpoint compatibility check guards against mixing models)."""
    sources = sorted((exp_dir / "config").glob("*.yaml"))
    sf_config = Path(sf_cfg.train_dir) / sf_cfg.experiment / "config.json"
    dest.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if not (dest / src.name).exists():
            shutil.copy2(src, dest / src.name)
    if sf_config.exists() and not (dest / "sf_config.json").exists():
        shutil.copy2(sf_config, dest / "sf_config.json")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


### Simulation ###


def flatten_info(info: dict[str, Any]) -> dict[str, float]:
    """Numeric scalar game variables; agent position from info['pos'] as pos_<key>."""
    flat: dict[str, float] = {}
    for key, val in info.items():
        if key == "pos" and isinstance(val, dict):
            for pkey, pval in val.items():
                if isinstance(pval, (int, float, np.number)) and not isinstance(
                    pval, bool
                ):
                    flat[f"pos_{pkey}"] = float(pval)
        elif isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
            flat[key] = float(val)
    return flat


def get_game_info(env) -> dict[str, float]:
    try:
        return flatten_info(env.unwrapped.get_info())
    except Exception:  # e.g. no game state available
        return {}


class Buffer:
    def __init__(self):
        self.data: dict[str, list[np.ndarray]] = {}

    def add(self, record: dict[str, np.ndarray]):
        for k, v in record.items():
            self.data.setdefault(k, []).append(v)

    def __len__(self):
        return len(self.data.get("steps/episode", []))

    def pop(self) -> dict[str, np.ndarray]:
        arrays = {k: np.stack(v) for k, v in self.data.items()}
        self.data = {}
        return arrays


class AsyncWriter:
    """Writes shards in a background thread (zlib releases the GIL, so compression overlaps with
    the simulation). At most one shard is in flight, which bounds memory use."""

    def __init__(self, store: Store, level: int):
        self.store = store
        self.level = level
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending: Future | None = None

    def wait(self):
        if self.pending is not None:
            pending, self.pending = self.pending, None
            pending.result()  # re-raises errors from the writer thread

    def submit(self, arrays: dict[str, np.ndarray], num_episodes_total: int):
        self.wait()
        self.pending = self.pool.submit(
            self.store.write_shard, arrays, num_episodes_total, self.level
        )

    def close(self):
        try:
            self.wait()
        finally:
            self.pool.shutdown()


def _to_storage_dtype(name: str, act: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.float16 and np.abs(act).max(initial=0) > np.finfo(np.float16).max:
        raise ValueError(
            f"Layer '{name}' exceeds the float16 range; use --dtype float32."
        )
    return act.astype(dtype)


def _schema_of(record: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in record.items()
    }


def main(argv: list[str] | None = None):  # noqa: C901
    args = parse_args(argv)
    exp_dir = args.experiment_dir.expanduser().resolve()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    act_dtype = np.dtype(args.dtype)

    cfg = load_experiment_cfg(exp_dir)
    train_env = cfg.dataset.env_name
    if args.env is not None:
        cfg.dataset.env_name = args.env
    env_name = cfg.dataset.env_name

    sf_cfg = build_sf_cfg(cfg, args, device_str)
    ckpt_path = resolve_checkpoint(sf_cfg, args.checkpoint)
    print(f"Experiment: {exp_dir}\nCheckpoint: {ckpt_path}\nEnvironment: {env_name}")

    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size
    from tqdm.auto import tqdm

    actor_critic, env, env_steps = make_agent(sf_cfg, ckpt_path, device)
    env_info = extract_env_info(env, sf_cfg)
    rnn_size = get_rnn_size(sf_cfg)

    def zero_rnn():
        return torch.zeros([1, rnn_size], dtype=torch.float32, device=device)

    seed = (
        args.seed
        if args.seed is not None
        else int(np.random.SeedSequence().entropy % 2**31)
    )
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)

    if args.list_layers:
        list_layers(actor_critic, obs, zero_rnn())
        env.close()
        return

    layers = resolve_layers(
        actor_critic.brain, args.layers, (actor_critic.actor, actor_critic.critic)
    )
    recorder = ActivationRecorder(actor_critic.brain, layers)

    output_dir = args.output_dir or exp_dir / "data" / "activations"
    name = args.name or f"{exp_dir.name}__{env_name}__{ckpt_path.stem}"
    store = Store((output_dir / name).expanduser().resolve(), args.overwrite)
    print(
        f"Store: {store.path} ({'new' if store.is_new else f'extending {store.num_frames} frames'})"
    )

    episode = store.next_episode
    episode_step = 0
    completed_episodes = 0
    rnn_states = zero_rnn()
    buffer = Buffer()
    writer = AsyncWriter(store, args.compression_level)
    info_keys: list[str] | None = store.manifest.get("info_keys")
    run_id = -1
    n = 0
    status = "completed"

    def should_stop() -> bool:
        if args.num_episodes is not None and completed_episodes >= args.num_episodes:
            return True
        if n < args.num_frames:
            return False
        return not (args.finish_episode and episode_step > 0)

    pbar = tqdm(total=args.num_frames, unit="frame")
    try:
        with torch.no_grad():
            while not should_stop():
                raw_input = obs["obs"][0].cpu().numpy().copy()
                measurements = (
                    obs["measurements"][0].cpu().numpy().astype(np.float32).copy()
                    if "measurements" in obs
                    else None
                )
                game_info = get_game_info(env)

                policy_outputs = actor_critic(
                    prepare_and_normalize_obs(actor_critic, obs), rnn_states
                )
                activations = recorder.pop()
                actions = policy_outputs["actions"]
                if args.deterministic:
                    actions = argmax_actions(actor_critic.action_distribution())
                if actions.ndim == 1:
                    actions = actions.unsqueeze(-1)
                rnn_states = policy_outputs["new_rnn_states"]

                obs, rew, terminated, truncated, _ = env.step(
                    preprocess_actions(env_info, actions)
                )
                term, trunc = bool(terminated[0]), bool(truncated[0])

                if info_keys is None:
                    info_keys = sorted(game_info)
                record: dict[str, np.ndarray] = {"input": raw_input}
                for spec in layers:
                    record[f"act/{spec.name}"] = _to_storage_dtype(
                        spec.name, activations[spec.name], act_dtype
                    )
                record.update(
                    {
                        "steps/run": np.int32(
                            run_id
                        ),  # set properly below on the first step
                        "steps/episode": np.int64(episode),
                        "steps/episode_step": np.int32(episode_step),
                        "steps/action": actions[0].cpu().numpy().astype(np.int64),
                        "steps/action_logits": policy_outputs["action_logits"][0]
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "steps/value": np.float32(
                            policy_outputs["values"].reshape(-1)[0].item()
                        ),
                        "steps/reward": np.float32(float(rew[0])),
                        "steps/terminated": np.bool_(term),
                        "steps/truncated": np.bool_(trunc),
                    }
                )
                if measurements is not None:
                    record["steps/measurements"] = measurements
                for key in info_keys:
                    record[f"steps/info/{key}"] = np.float32(game_info.get(key, np.nan))

                if run_id < 0:  # first step: we now know all shapes
                    schema = {
                        "experiment_dir": str(exp_dir),
                        "train_env": train_env,
                        "brain": OmegaConf.to_container(cfg.brain, resolve=True),
                        "transforms": OmegaConf.to_container(
                            OmegaConf.create(cfg.dataset.get("transforms", [])),
                            resolve=True,
                        ),
                        "input": _schema_of({"input": raw_input})["input"],
                        "layers": _schema_of(
                            {s.name: record[f"act/{s.name}"] for s in layers}
                        ),
                        "steps": _schema_of(
                            {
                                k[len("steps/") :]: v
                                for k, v in record.items()
                                if k.startswith("steps/")
                            }
                        ),
                        "info_keys": info_keys,
                    }
                    model = {"checkpoint": str(ckpt_path), "env_steps": env_steps}
                    store.init_schema(schema, model, args.allow_model_mismatch)
                    run_id = store.start_run(
                        {
                            "env": env_name,
                            "checkpoint": str(ckpt_path),
                            "checkpoint_env_steps": env_steps,
                            "seed": seed,
                            "deterministic": args.deterministic,
                            "frameskip": int(sf_cfg.env_frameskip),
                            "device": device_str,
                            "activation_dtype": args.dtype,
                            "command": " ".join(sys.argv),
                            "git_commit": _git_commit(),
                            "host": socket.gethostname(),
                        }
                    )
                    record["steps/run"] = np.int32(run_id)
                    copy_configs(exp_dir, sf_cfg, store.path / "config")
                    bytes_per_frame = sum(v.nbytes for v in record.values())
                    print(
                        f"Uncompressed size per frame: {bytes_per_frame / 1024:.1f} KiB"
                    )

                buffer.add(record)
                n += 1
                pbar.update(1)

                if term or trunc:
                    episode += 1
                    episode_step = 0
                    completed_episodes += 1
                    rnn_states = (
                        zero_rnn()
                    )  # env auto-resets; obs is the new episode's first frame
                else:
                    episode_step += 1

                episode_ended = term or trunc
                if len(buffer) >= args.max_shard_size or (
                    episode_ended and len(buffer) >= args.shard_size
                ):
                    writer.submit(buffer.pop(), episode + (episode_step > 0))
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nInterrupted, saving what was recorded so far...")
    except Exception:
        status = "failed"
        raise
    finally:
        pbar.close()
        recorder.remove()
        if run_id >= 0:
            if len(buffer) > 0:
                writer.submit(buffer.pop(), episode + (episode_step > 0))
            writer.close()
            store.finish_run(status, completed_episodes=completed_episodes)
            print(
                f"Recorded {n} frames ({completed_episodes} completed episodes) in run {run_id}. "
                f"Store now holds {store.num_frames} frames: {store.path}"
            )
        env.close()


if __name__ == "__main__":
    main()
