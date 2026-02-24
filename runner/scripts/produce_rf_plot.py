import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import yaml
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm


def rescale(x: npt.NDArray[Any], min: float = 0, max: float = 1) -> npt.NDArray[Any]:
    _max = np.max(np.abs(x))
    _min = -_max  # keep 0 as the central value
    return ((x - _min) / (_max - _min)) * (max - min) + min


def reshape_images(
    arr: npt.NDArray[Any],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    whitespace: float = 0.1,
    rescale_individ: bool = False,
):
    n, _, w, h = arr.shape
    whitespace_pix = np.round(whitespace * max(w, h)).astype(int)
    if n_rows is None:
        n_rows = (n + n_cols - 1) // n_cols if n_cols is not None else 1
    if n_cols is None:
        n_cols = (n + n_rows - 1) // n_rows

    # Calculate the total width and height of the final image
    total_width = n_cols * w + (n_cols - 1) * whitespace_pix
    total_height = n_rows * h + (n_rows - 1) * whitespace_pix

    # Create a new image with the calculated dimensions
    final_image = np.full(
        (3, total_height, total_width),
        1 if rescale_individ else arr.max(),
        dtype=np.float32,
    )

    # Populate the final image with the individual images
    for i in range(n):
        row = i // n_cols
        col = i % n_cols
        x1 = col * (w + whitespace_pix)
        y1 = row * (h + whitespace_pix)
        x2 = x1 + w
        y2 = y1 + h
        final_image[:, y1:y2, x1:x2] = rescale(arr[i]) if rescale_individ else arr[i]

    if not rescale_individ:
        final_image = rescale(final_image)

    return np.moveaxis(final_image, 0, -1)


def get_rf_files(rf_dir: Path) -> list[str]:
    rf_files = os.listdir(rf_dir)
    rf_files.sort(key=lambda f: os.path.getctime(rf_dir / f))
    return rf_files


def init_plot(
    rf_dir: Path, cur_file: str, hyper_params: list[str], figwidth: float = 10
):
    # Init figure
    if cur_file.endswith(".json"):
        with open(rf_dir / cur_file) as f:
            rf = json.load(f)
    else:
        rf = np.load(rf_dir / cur_file, allow_pickle=True)

    comp_layer_rfs = []
    for i, (layer, layer_rfs) in enumerate(rf.items()):
        comp_layer_rfs.append(
            reshape_images(np.array(layer_rfs), n_cols=8, whitespace=0.1)
        )

    height_ratios = [x.shape[0] / x.shape[1] for x in comp_layer_rfs]

    fig, _ = plt.subplots(
        nrows=len(rf.keys()),
        ncols=1,
        height_ratios=height_ratios,
        figsize=(figwidth, figwidth * sum(height_ratios) + 1),
    )
    imshows = []
    for i, (layer, layer_rfs) in enumerate(zip(rf.keys(), comp_layer_rfs)):
        plt.subplot(len(rf.keys()), 1, i + 1)
        imshows.append(plt.imshow(layer_rfs))
        plt.axis("off")
        plt.title(layer, loc="left")
    suptitle = plt.suptitle(get_title(hyper_params, cur_file))
    plt.tight_layout()

    return fig, imshows, suptitle


def get_hyperparams(experiment_path: Path) -> list[str]:
    config_path = experiment_path / "config/config.yaml"
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return [
        config["env_name"][10:],
        "rnn" if "rnn" in config["brain"]["circuits"] else "feedforward",
        "weight=" + str(config["recon_weight"]),
    ]


def get_title(hyper_params: list[str], cur_file: str) -> str:
    return str.join(", ", [*hyper_params, "step=" + cur_file.split("_")[-1][:-5]])


def produce_image(experiment_path: Path, out_dir: Path, last: bool = True):
    rf_dir = experiment_path / "data/analyses"
    index = -1 if last else 3

    rf_files = get_rf_files(rf_dir)
    hyper_params = get_hyperparams(experiment_path)
    init_plot(rf_dir, rf_files[index], hyper_params)
    filename = str.join("_", hyper_params) + ("_last" if last else "_early") + ".png"
    plt.savefig(out_dir / filename)
    plt.close()


def produce_anim(experiment_path: Path, out_dir: Path, fast: bool = False):
    rf_dir = experiment_path / "data/analyses"
    rf_files = get_rf_files(rf_dir)
    hyper_params = get_hyperparams(experiment_path)

    fig, imshows, suptitle = init_plot(
        rf_dir, rf_files[0], hyper_params, figwidth=5 if fast else 10
    )

    step = 3 if fast else 1
    n_frames = len(rf_files) // step

    def update(frame: int):
        cur_file = rf_files[min(frame * step, len(rf_files) - 1)]
        with open(rf_dir / cur_file) as f:
            rf = json.load(f)

        for i, (_, layer_rfs) in enumerate(rf.items()):
            imshows[i].set_array(
                reshape_images(np.array(layer_rfs), n_cols=8, whitespace=0.1)
            )
        suptitle.set_text(get_title(hyper_params, cur_file))
        return [*imshows, suptitle]

    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=100)
    plt.close()

    class ProgressBar(tqdm):
        def update_to(self, current: int, total: int):
            self.total = total
            self.update(current - self.n)

    progress = ProgressBar()

    filename = str.join("_", hyper_params) + "_anim.gif"
    ani.save(
        filename=out_dir / filename,
        writer="pillow",
        progress_callback=progress.update_to,
    )


def parse_args(argv: list[str]):
    experiments_path = Path(argv[1])
    out_dir = Path(argv[2])
    anim = False
    fast = False
    if len(argv) > 3:
        anim = sys.argv[3] == "--anim"
    if len(argv) > 4:
        fast = sys.argv[4] == "--fast"
    return experiments_path, out_dir, anim, fast


experiments_path, out_dir, anim, fast = parse_args(sys.argv)
if (experiments_path / "data").exists():
    _iter = [experiments_path]
else:
    _iter = experiments_path.iterdir()
for experiment_path in _iter:
    try:
        print(experiment_path)
        if anim:
            produce_anim(experiment_path, out_dir, fast)
        produce_image(experiment_path, out_dir, last=True)
        produce_image(experiment_path, out_dir, last=False)
    except Exception as e:
        print(f"Error processing {experiment_path}: {e}")
        continue
