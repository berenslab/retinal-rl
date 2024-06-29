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
    nclrs, hght, wdth = enc.input_shape
    ochns = nclrs

    imgsz = [1, nclrs, hght, wdth]
    # Obs requires grad
    obs = torch.zeros(size=imgsz, device=device, requires_grad=True)

    stas: Dict[str, NDArray[np.float64]] = {}

    mdls: List[nn.Module] = []

    with torch.no_grad():
        for lyrnm, mdl in enc.conv_head.named_children():
            gradient_calculator = NeuronGradient(enc, mdl)
            mdls.append(mdl)

            # check if mdl has out channels
            if hasattr(mdl, "out_channels"):
                ochns = mdl.out_channels
            hsz, wsz = encoder_out_size(mdls, hght, wdth)

            hidx = (hsz - 1) // 2
            widx = (wsz - 1) // 2

            hrf_size, wrf_size, hmn, wmn = rf_size_and_start(mdls, hidx, widx)

            hmx = hmn + hrf_size
            wmx = wmn + wrf_size

            stas[lyrnm] = np.zeros((ochns, nclrs, hrf_size, wrf_size))

            for j in range(ochns):
                grad = (
                    gradient_calculator.attribute(obs, (j, hidx, widx))[
                        0, :, hmn:hmx, wmn:wmx
                    ]
                    .cpu()
                    .numpy()
                )

                stas[lyrnm][j] = grad

    return stas


def get_reconstructions(
    device: torch.device,
    brain: Brain,
    dataset: Dataset[Tuple[Tensor, int]],
    sample_size: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    brain.eval()  # Set the model to evaluation mode
    original_images: List[Tensor] = []
    reconstructed_images: List[Tensor] = []
    indices = torch.randperm(len(dataset))[:sample_size]

    with torch.no_grad():  # Disable gradient computation
        for index in indices:
            img, _ = dataset[index]
            img = img.to(device)
            stimulus = {"vision": img.unsqueeze(0)}
            recimg = brain(stimulus)["decoder"].squeeze(0)
            original_images.append(img.cpu())
            reconstructed_images.append(recimg.cpu())

    return original_images, reconstructed_images
