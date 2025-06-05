import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import yaml
from matplotlib import pyplot as plt


def rescale(x: npt.NDArray[Any], min: float = 0, max: float = 1) -> npt.NDArray[Any]:
    return ((x - x.min()) / (x.max() - x.min())) * (max - min) + min


def reshape_images(
    arr: npt.NDArray[Any],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    whitespace: float = 0.1,
    rescale_individ: bool = False,
):
    n, _, w, h = arr.shape
    whitespace_pix = np.round(whitespace * max(w, h)).astype(int)
    if n_rows is None and n_cols is None:
        n_rows = 1
    if n_rows is None:
        n_rows = (n + n_cols - 1) // n_cols
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

def produce_image(experiment_path: Path, out_dir: Path, last= True):
    rf_dir = experiment_path / "data/analyses"
    config_path = experiment_path / "config/config.yaml"

    index = -1 if last else 3

    with open(config_path) as file:
        config = yaml.safe_load(file)

    rf_files = os.listdir(rf_dir)
    rf_files.sort(key= lambda f: os.path.getctime(rf_dir / f))

    cur_file = rf_files[index]
    with open (rf_dir / cur_file) as f:
        rf = json.load(f)

    hyper_params = [
        config["env_name"][10:],
        "rnn" if "rnn" in config["brain"]["circuits"] else "feedforward",
        "weight=" + str(config["recon_weight"]),
        "step=" + cur_file.split("_")[-1][:-5],
    ]
    comp_layer_rfs = []
    for i, (layer, layer_rfs) in enumerate(rf.items()):
        comp_layer_rfs.append(reshape_images(np.array(layer_rfs), n_cols=8, whitespace=0.1))

    height_ratios = [x.shape[0]/x.shape[1] for x in comp_layer_rfs]

    fig = plt.subplots(nrows=len(rf.keys()), ncols=1, height_ratios=height_ratios, figsize=(10, 10*sum(height_ratios)+1))
    for i, (layer, layer_rfs) in enumerate(zip(rf.keys(), comp_layer_rfs)):
        plt.subplot(len(rf.keys()), 1, i + 1)
        plt.imshow(layer_rfs)
        plt.axis("off")
        plt.title(layer, loc='left')
    plt.suptitle(str.join(", ", hyper_params))
    plt.tight_layout()
    filename = str.join("_", hyper_params[:-1])+ ("_last" if last else "_early")+".png"
    plt.savefig(out_dir / filename)
    plt.close()

experiment_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

produce_image(experiment_path, out_dir, last=True)
produce_image(experiment_path, out_dir, last=False)