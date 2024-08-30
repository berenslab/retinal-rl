from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from captum.attr import NeuronGradient
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from retinal_rl.models.brain import Brain
from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder
from retinal_rl.models.util import encoder_out_size, rf_size_and_start


def gradient_receptive_fields(
    device: torch.device, enc: ConvolutionalEncoder
) -> Dict[str, NDArray[np.float64]]:
    """Return the receptive fields of every layer of a convnet as computed by neural gradients.

    The returned dictionary is indexed by layer name and the shape of the array
    (#layer_channels, #input_channels, #rf_height, #rf_width).
    """
    enc.eval()
    nclrs, hght, wdth = enc.input_shape
    out_channels = nclrs

    imgsz = [1, nclrs, hght, wdth]
    # Obs requires grad
    obs = torch.zeros(size=imgsz, device=device, requires_grad=True)

    stas: Dict[str, NDArray[np.float64]] = {}

    mdls: List[nn.Module] = []

    with torch.no_grad():
        for layer_name, mdl in enc.conv_head.named_children():
            gradient_calculator = NeuronGradient(enc, mdl)
            mdls.append(mdl)

            # check if mdl has out channels
            if hasattr(mdl, "out_channels"):
                out_channels = mdl.out_channels
            hsz, wsz = encoder_out_size(mdls, hght, wdth)

            hidx = (hsz - 1) // 2
            widx = (wsz - 1) // 2

            hrf_size, wrf_size, h_min, w_min = rf_size_and_start(mdls, hidx, widx)

            # Assert min max is in bounds
            # potential TODO: change input size if rf is larger than actual input
            h_min = max(0,h_min)
            w_min = max(0,w_min)
            hrf_size = min(hght,hrf_size)
            hrf_size = min(wdth,hrf_size)

            h_max = h_min + hrf_size
            w_max = w_min + wrf_size

            stas[layer_name] = np.zeros((out_channels, nclrs, hrf_size, wrf_size))
            for j in range(out_channels):
                grad = (
                    gradient_calculator.attribute(obs, (j, hidx, widx))[
                        0, :, h_min:h_max, w_min:w_max
                    ]
                    .cpu()
                    .numpy()
                )

                stas[layer_name][j] = grad

    return stas


def get_reconstructions(
    device: torch.device,
    brain: Brain,
    test_set: Dataset[Tuple[Tensor, int]],
    train_set: Dataset[Tuple[Tensor, int]],
    sample_size: int,
) -> Dict[str, List[Tuple[Tensor, int]]]:
    brain.eval()  # Set the model to evaluation mode

    def collect_reconstructions(
        data_set: Dataset[Tuple[Tensor, int]], sample_size: int
    ) -> Tuple[List[Tuple[Tensor, int]], List[Tuple[Tensor, int]]]:
        subset: List[Tuple[Tensor, int]] = []
        estimates: List[Tuple[Tensor, int]] = []
        indices = torch.randperm(len(data_set))[:sample_size]

        with torch.no_grad():  # Disable gradient computation
            for index in indices:
                img, k = data_set[index]
                img = img.to(device)
                stimulus = {"vision": img.unsqueeze(0)}
                response = brain(stimulus)
                rec_img = response["decoder"].squeeze(0)
                pred_k = response["classifier"].argmax().item()
                subset.append((img.cpu(), k))
                estimates.append((rec_img.cpu(), pred_k))

        return subset, estimates

    train_subset, train_estimates = collect_reconstructions(train_set, sample_size)
    test_subset, test_estimates = collect_reconstructions(test_set, sample_size)

    return {
        "train_subset": train_subset,
        "train_estimates": train_estimates,
        "test_subset": test_subset,
        "test_estimates": test_estimates,
    }
