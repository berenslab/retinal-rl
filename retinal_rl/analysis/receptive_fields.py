import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor, nn

from retinal_rl.analysis.plot import FigureLogger
from retinal_rl.models.brain import Brain, get_cnn_circuit
from retinal_rl.util import FloatArray, is_nonlinearity, rf_size_and_start


def analyze(brain: Brain, device: torch.device):
    brain.eval()
    brain.to(device)
    input_shape, cnn_layers = get_cnn_circuit(brain)

    # Analyze each layer
    head_layers: list[
        nn.Module
    ] = []  # possible TODO: have get cnn circuit return a nn.Sequential, then looping here is nicer / easier

    results: dict[str, FloatArray] = {}
    for layer_name, layer in cnn_layers.items():
        head_layers.append(layer)

        if is_nonlinearity(layer):
            continue

        if isinstance(layer, nn.Conv2d):
            out_channels = layer.out_channels
        else:
            raise NotImplementedError(
                "Can only compute receptive fields for 2d convolutional layers"
            )

        results[layer_name] = _compute_receptive_fields(
            device, head_layers, input_shape, out_channels
        )

    return input_shape, results


def plot(
    log: FigureLogger,
    rf_result: dict[str, FloatArray],
    epoch: int,
    copy_checkpoint: bool,
):
    for layer_name, layer_rfs in rf_result.items():
        layer_rf_plots = layer_receptive_field_plots(layer_rfs)
        log.log_figure(
            layer_rf_plots,
            "receptive_fields",
            f"{layer_name}",
            epoch,
            copy_checkpoint,
        )


def layer_receptive_field_plots(lyr_rfs: FloatArray, max_cols: int = 8) -> Figure:
    """Plot the receptive fields of a convolutional layer."""
    ochns, _, _, _ = lyr_rfs.shape

    # Calculate the number of rows needed based on max_cols
    cols = min(ochns, max_cols)
    rows = ochns // cols + (1 if ochns % cols > 0 else 0)

    fig, axs0 = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2, 1.6 * rows),
        squeeze=False,
    )

    axs: list[Axes] = axs0.flat

    for i in range(ochns):
        ax = axs[i]
        data = np.moveaxis(lyr_rfs[i], 0, -1)  # Move channel axis to the last dimension
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) / (data_max - data_min)
        ax.imshow(data)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title(f"Channel {i+1}")

    fig.tight_layout()  # Adjust layout to fit color bars
    return fig


def _compute_receptive_fields(
    device: torch.device,
    head_layers: list[nn.Module],
    input_shape: tuple[int, ...],
    out_channels: int,
) -> FloatArray:
    """Compute receptive fields for a sequence of layers."""
    nclrs, hght, wdth = input_shape
    imgsz = [1, nclrs, hght, wdth]
    obs = torch.zeros(size=imgsz, device=device, requires_grad=True)

    head_model = nn.Sequential(*head_layers)
    x = head_model(obs)

    hsz, wsz = x.shape[2:]
    hidx = (hsz - 1) // 2
    widx = (wsz - 1) // 2

    hrf_size, wrf_size, hmn, wmn = rf_size_and_start(head_layers, hidx, widx)
    grads: list[Tensor] = []

    for j in range(out_channels):
        grad = torch.autograd.grad(x[0, j, hidx, widx], obs, retain_graph=True)[0]
        grads.append(grad[0, :, hmn : hmn + hrf_size, wmn : wmn + wrf_size])

    return torch.stack(grads).cpu().numpy()
