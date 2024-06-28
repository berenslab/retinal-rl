### Util for preparing simulations and data for analysis

from math import ceil, floor
from os.path import join

import numpy as np
import torch
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir
from torch import nn

## Paths ###

resources_dir = "resources"


def analysis_root(cfg):
    """Returns the root analysis directory."""
    return join(experiment_dir(cfg), "analyses")


def analysis_path(cfg, ana_name):
    """Returns the path to the analysis directory."""
    art = analysis_root(cfg)

    return join(art, ana_name)


# Write number of analyses to file
def write_analysis_count(cfg, num):
    """Writes the number of analyses to file, replacing the old file."""
    art = analysis_root(cfg)
    with open(join(art, "analysis_count.txt"), "w") as f:
        f.write(str(num))


# Read number of analyses from file
def read_analysis_count(cfg):
    """Reads the number of analyses from file; returns 0 if file doesn't exist."""
    art = analysis_root(cfg)
    try:
        with open(join(art, "analysis_count.txt"), "r") as f:
            return int(f.read())
    except:
        return 0


#
# def get_analysis_times(cfg):
#     """
#     Returns the list of analysis times.
#     """
#     art = analysis_root(cfg)
#     # list directories
#     drs = os.listdir(art)
#     # filter out directories that don't start with "env_steps-"
#     drs = [d for d in drs if d.startswith("env_steps-")]
#
#     return [int(f.split("-")[1]) for f in drs]
#
def data_path(cfg, ana_name, flnm=None):
    """Returns the path to the data directory."""
    datpth = analysis_path(cfg, ana_name) + "/data"

    if flnm is not None:
        datpth = datpth + "/" + flnm

    return datpth


def plot_path(cfg, ana_name, flnm=None):
    """Returns the path to the plot directory."""
    pltpth = analysis_path(cfg, ana_name) + "/plots"

    if flnm is not None:
        pltpth = pltpth + "/" + flnm

    return pltpth


### IO ###


def save_onnx(cfg: Config, ana_name: str, brain, inpts) -> None:
    """Write an onnx file of the saved model."""
    # Note that onnx can't process dictionary inputs and so we can only look at the encoder (and decoder?) separately)
    torch.onnx.export(
        brain.valnet,
        inpts,
        data_path(cfg, ana_name, "value_network.onnx"),
        verbose=False,
        input_names=["observation", "measurements", "rnn_states"],
        output_names=["value"],
    )


def save_data(cfg: Config, ana_name, dat, flnm):
    """Saves data. 'dat' should probably be a dictionary."""
    np.save(data_path(cfg, ana_name, flnm), dat, allow_pickle=True)


def load_data(cfg: Config, ana_name, flnm):
    """Loads data. Note the use of tolist() is necessary to read dictionaries."""
    return np.load(data_path(cfg, ana_name, flnm) + ".npy", allow_pickle=True).tolist()


### Misc analysis tools ###


def normalize(x, min=0, max=1):
    return (max - min) * (x - np.min(x)) / (np.max(x) - np.min(x)) + min


def from_float_to_rgb(xs):
    return (255 * normalize(xs)).astype(np.uint8)


def obs_dict_to_tuple(obs_dct):
    """Extract observation"""
    obs = obs_dct["obs"][0]
    msm = None
    if "measurements" in obs_dct.keys():
        msm = obs_dct["measurements"][0]
    # visualize obs only for the 1st agent
    return (obs, msm)


def obs_to_img(obs):
    """Rearrange an image so it can be presented by matplot lib."""
    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    img = obs.cpu().numpy()
    return img


### Network Tools ###


def activation(act) -> nn.Module:
    if act == "elu":
        return nn.ELU(inplace=True)
    elif act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "tanh":
        return nn.Tanh()
    elif act == "softplus":
        return nn.Softplus()
    elif act == "identity":
        return nn.Identity(inplace=True)
    else:
        raise Exception("Unknown activation function")


def is_activation(mdl: nn.Module) -> bool:
    bl = any(
        [
            isinstance(mdl, nn.ELU),
            isinstance(mdl, nn.ReLU),
            isinstance(mdl, nn.Tanh),
            isinstance(mdl, nn.Softplus),
            isinstance(mdl, nn.Identity),
        ]
    )
    return bl


def double_up(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return x


def encoder_out_size(mdls, hght0, wdth0):
    """Compute the size of the encoder output, where mdls is the list of encoder
    modules.
    """
    hght = hght0
    wdth = wdth0

    # iterate over modules that are not activations
    for mdl in mdls:
        if is_activation(mdl):
            continue

        krnsz = double_up(mdl.kernel_size)
        strd = double_up(mdl.stride)
        pad = double_up(mdl.padding)

        # if has a ceil mode
        if hasattr(mdl, "ceil_mode") and mdl.ceil_mode:
            hght = ceil((hght - krnsz[0] + 2 * pad[0]) / strd[0] + 1)
            wdth = ceil((wdth - krnsz[1] + 2 * pad[1]) / strd[1] + 1)
        else:
            hght = floor((hght - krnsz[0] + 2 * pad[0]) / strd[0] + 1)
            wdth = floor((wdth - krnsz[1] + 2 * pad[1]) / strd[1] + 1)

    return hght, wdth


def padder(krnsz):
    return (krnsz - 1) // 2


def fill_in_argv_template(argv):
    """Replace string templates in argv with values from argv."""
    # If calling for help (-h or --help), don't replace anything
    if any([a in argv for a in ["-h", "--help"]]):
        return argv

    # Convert argv into a dictionary
    argv = [a.split("=") for a in argv]
    # Remove dashes from argv
    cfg = dict([[a[0].replace("--", ""), a[1]] for a in argv])
    # Replace cfg string templates
    cfg = {k: v.format(**cfg) for k, v in cfg.items()}
    # Convert cfg back into argv
    argv = [f"--{k}={v}" for k, v in cfg.items()]

    return argv
