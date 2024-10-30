import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from captum.attr import NeuronGradient
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import deprecated

from retinal_rl.models.brain import Brain
from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder
from retinal_rl.util import encoder_out_size, rf_size_and_start


@deprecated("Use functions of retinal_rl.analysis.statistics")
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
            h_min = max(0, h_min)
            w_min = max(0, w_min)
            hrf_size = min(hght, hrf_size)
            wrf_size = min(wdth, wrf_size)

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


def _activation_triggered_average(
    model: nn.Sequential, n_batch: int = 2048, rf_size=None, device=None
):
    model.eval()
    if rf_size is None:
        _out_channels, input_size = get_input_output_shape(model)
    else:
        input_size = rf_size
    input_tensor = torch.randn(
        (n_batch, *input_size), requires_grad=False, device=device
    )
    output = model(input_tensor)
    output = sum_collapse_output(output)
    input_tensor = input_tensor[:, None, :, :, :].expand(
        -1, output.shape[1], -1, -1, -1
    )

    weights = output[:, :, None, None, None].expand(-1, -1, *input_size)
    weight_sums = output.sum(0)
    weight_sums[weight_sums == 0] = 1
    weighted = (weights * input_tensor).sum(0)
    return weighted.cpu().detach(), weight_sums.cpu().detach()


def activation_triggered_average(
    model: nn.Sequential,
    n_batch: int = 2048,
    n_iter: int = 1,
    rf_size=None,
    device=None,
) -> Dict[str, NDArray[np.float64]]:
    # TODO: WIP
    warnings.warn("Code is not tested and might contain bugs.")
    stas: Dict[str, NDArray[np.float64]] = {}
    with torch.no_grad():
        for index, (layer_name, mdl) in tqdm(
            enumerate(model.named_children()), total=len(model)
        ):
            weighted, weight_sums = _activation_triggered_average(
                model[: index + 1], n_batch, device=device
            )
            for _ in tqdm(range(n_iter - 1), total=n_iter - 1, leave=False):
                it_weighted, it_weight_sums = _activation_triggered_average(
                    model[: index + 1], n_batch, rf_size, device=device
                )
                weighted += it_weighted
                weight_sums += it_weight_sums
            stas[layer_name] = (
                weighted.cpu().detach()
                / weight_sums[:, None, None, None]
                / len(weight_sums)
            ).numpy()
        torch.cuda.empty_cache()
    return stas


@deprecated("Use functions of retinal_rl.analysis.statistics")
def sum_collapse_output(out_tensor):
    if len(out_tensor.shape) > 2:
        sum_dims = [2 + i for i in range(len(out_tensor.shape) - 2)]
        out_tensor = torch.sum(out_tensor, dim=sum_dims)
    return out_tensor


def _find_last_layer_shape(
    model: nn.Sequential,
) -> Tuple[int, Optional[int], Optional[int], Optional[int], bool]:
    _first = 0
    down_stream_linear = False
    num_outputs = None
    in_size, in_channels = None, None
    for i, layer in enumerate(reversed(model)):
        _first += 1
        if isinstance(layer, nn.Linear):
            num_outputs = layer.out_features
            in_channels = 1
            in_size = layer.in_features
            down_stream_linear = True
            break
        if isinstance(layer, nn.Conv2d):
            num_outputs = layer.out_channels
            in_channels = layer.in_channels
            in_size = (
                layer.in_channels
                * ((layer.kernel_size[0] - 1) * layer.dilation[0] + 1) ** 2
            )
            break
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            for prev_layer in reversed(model[: -i - 1]):
                if isinstance(prev_layer, nn.Conv2d):
                    in_channels = prev_layer.out_channels
                    break
                if isinstance(prev_layer, nn.Linear):
                    in_channels = 1
                else:
                    raise TypeError("layer before pooling needs to be conv or linear")
            _kernel_size = (
                layer.kernel_size
                if isinstance(layer.kernel_size, int)
                else layer.kernel_size[0]
            )
            in_size = _kernel_size**2 * in_channels
            break
    return _first, num_outputs, in_size, in_channels, down_stream_linear


@deprecated("Use functions of retinal_rl.analysis.statistics")
def get_input_output_shape(model: nn.Sequential):
    """
    Calculates the 'minimal' input and output of a sequential model.
    If last layer is a convolutional layer, output is assumed to be the number of channels (so 1x1 in space).
    Takes into account if last layer is a pooling layer.
    For linear layer obviously the number of out_features.
    TODO: assert kernel sizes etc are quadratic / implement adaptation to non quadratic kernels
    TODO: Check if still needed, function near duplicate of some of Sachas code
    """

    _first, num_outputs, in_size, in_channels, down_stream_linear = (
        _find_last_layer_shape(model)
    )

    for i, layer in enumerate(reversed(model[:-_first])):
        if isinstance(layer, nn.Linear):
            if num_outputs is None:
                num_outputs = layer.out_features
            in_channels = 1
            in_size = layer.in_features
            down_stream_linear = True
        elif isinstance(layer, nn.Conv2d):
            if num_outputs is None:
                num_outputs = layer.out_channels
            in_size = math.sqrt(in_size / in_channels)
            in_channels = layer.in_channels
            in_size = (
                (in_size - 1) * layer.stride[0]
                - 2 * layer.padding[0] * down_stream_linear
                + ((layer.kernel_size[0] - 1) * layer.dilation[0] + 1)
            )
            in_size = in_size**2 * in_channels
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            for prev_layer in reversed(model[: -i - _first - 1]):
                if isinstance(prev_layer, nn.Conv2d):
                    in_channels = prev_layer.out_channels
                    break
            in_size = math.sqrt(in_size / in_channels)
            in_size = (
                (in_size - 1) * layer.stride[0]
                - 2 * layer.padding[0] * down_stream_linear
                + layer.kernel_size
            )
            in_size = in_size**2 * in_channels

    in_size = math.floor(math.sqrt(in_size / in_channels))
    input_size = (in_channels, in_size, in_size)
    return num_outputs, input_size


@deprecated("Use functions of retinal_rl.analysis.statistics")
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
