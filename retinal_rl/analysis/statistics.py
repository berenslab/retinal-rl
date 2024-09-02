import logging
from typing import Dict, Iterator, List, OrderedDict, Tuple, cast

import networkx as nx
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from retinal_rl.models.brain import Brain
from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder
from retinal_rl.util import FloatArray, is_activation, rf_size_and_start

logger = logging.getLogger(__name__)


def reconstruct_images(
    device: torch.device,
    brain: Brain,
    test_set: Dataset[Tuple[Tensor, int]],
    train_set: Dataset[Tuple[Tensor, int]],
    sample_size: int,
) -> Dict[str, List[Tuple[Tensor, int]]]:
    """Reconstruct images from a dataset using a trained Brain."""
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


def cnn_linear_receptive_fields(
    device: torch.device,
    input_shape: Tuple[int, ...],
    layers: OrderedDict[str, nn.Module],
) -> Dict[str, FloatArray]:
    """Return the receptive fields of every layer of a convnet as computed by neural gradients.

    The returned dictionary is indexed by layer name and the shape of the array
    (#layer_channels, #input_channels, #rf_height, #rf_width).
    """
    nclrs, hght, wdth = input_shape

    imgsz = [1, nclrs, hght, wdth]
    obs = torch.zeros(size=imgsz, device=device, requires_grad=True)

    stas: Dict[str, FloatArray] = {}

    # Create a sequential model from the layers
    model = nn.Sequential(layers)
    model.to(device)
    model.eval()

    hmn: int = 0
    hmx: int = 0

    mdls: List[nn.Module] = []

    for layer_name, layer in layers.items():
        # Forward pass through all preceding layers
        mdls.append(layer)
        x = obs
        for prev_layer in model:
            x = prev_layer(x)
            if prev_layer == layer:
                break

        if not hasattr(layer, "out_channels"):
            continue  # Skip layers without out_channels (e.g., activation functions)

        ochns = layer.out_channels
        hsz, wsz = x.shape[2:]  # Get the current height and width

        hidx = (hsz - 1) // 2
        widx = (wsz - 1) // 2

        hrf_size, wrf_size, hmn, wmn = rf_size_and_start(mdls, hidx, widx)

        hmx = hmn + hrf_size
        wmx = wmn + wrf_size

        stas[layer_name] = np.zeros((ochns, nclrs, hrf_size, wrf_size))

        for j in range(ochns):
            x.retain_grad()
            y = x[0, j, hidx, widx]
            y.backward(retain_graph=True)

            grad = obs.grad[0, :, hmn:hmx, wmn:wmx].cpu().numpy()
            stas[layer_name][j] = grad

            obs.grad.zero_()

    return stas


def layer_pixel_histograms(
    device: torch.device, dataloader: DataLoader[Tuple[Tensor, int]], num_bins: int = 20
) -> Dict[str, FloatArray]:
    """Compute histograms of pixel/activation values for each channel across all data in a dataset."""
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)
    num_channels: int = first_batch.shape[1]

    # Initialize variables for dynamic range computation
    global_min = torch.full((num_channels,), float("inf"), device=device)
    global_max = torch.full((num_channels,), float("-inf"), device=device)

    # First pass: compute global min and max
    total_elements = 0

    for batch, _ in dataloader:
        batch = batch.to(device)
        batch_min, _ = batch.view(-1, num_channels).min(dim=0)
        batch_max, _ = batch.view(-1, num_channels).max(dim=0)
        global_min = torch.min(global_min, batch_min)
        global_max = torch.max(global_max, batch_max)
        total_elements += batch.numel() // num_channels

    # Compute histogram parameters
    hist_range: tuple[float, float] = (global_min.min().item(), global_max.max().item())

    histograms: Tensor = torch.zeros(
        (num_channels, num_bins), dtype=torch.float64, device=device
    )

    for batch, _ in dataloader:
        batch = batch.to(device)
        for c in range(num_channels):
            channel_data = batch[:, c, :, :].reshape(-1)
            hist = torch.histc(
                channel_data, bins=num_bins, min=hist_range[0], max=hist_range[1]
            )
            histograms[c] += hist

    bin_width = (hist_range[1] - hist_range[0]) / num_bins
    normalized_histograms = histograms / (total_elements * bin_width / num_channels)

    return {
        "bin_edges": np.linspace(
            hist_range[0], hist_range[1], num_bins + 1, dtype=np.float64
        ),
        "channel_histograms": normalized_histograms.cpu().numpy(),
    }


def layer_spectral_analysis(
    device: torch.device,
    dataloader: DataLoader[Tuple[Tensor, int]],
) -> Dict[str, FloatArray]:
    """Compute power spectrum statistics for each channel across all data in a dataset."""
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)
    image_size = first_batch.shape[1:]

    # Initialize variables for dynamic range computation
    mean_power_spectrum = torch.zeros(image_size, dtype=torch.float64, device=device)
    m2_power_spectrum = torch.zeros(image_size, dtype=torch.float64, device=device)
    mean_autocorr = torch.zeros(image_size, dtype=torch.float64, device=device)
    m2_autocorr = torch.zeros(image_size, dtype=torch.float64, device=device)
    autocorr: Tensor = torch.zeros(image_size, dtype=torch.float64, device=device)

    count = 0

    for batch, _ in dataloader:
        batch = batch.to(device)
        for image in batch:
            count += 1

            # Compute power spectrum
            power_spectrum = torch.abs(fft.fft2(image)) ** 2

            # Compute power spectrum statistics
            mean_power_spectrum += power_spectrum
            m2_power_spectrum += power_spectrum**2

            # Compute normalized autocorrelation
            autocorr = cast(Tensor, fft.ifft2(power_spectrum)).real
            max_abs_autocorr = torch.amax(torch.abs(autocorr), dim=(-2, -1), keepdim=True)
            autocorr = autocorr / (max_abs_autocorr + 1e-8)

            # Compute autocorrelation statistics
            mean_autocorr += autocorr
            m2_autocorr += autocorr**2

    mean_power_spectrum /= count
    mean_autocorr /= count
    var_power_spectrum = m2_power_spectrum / count - (mean_power_spectrum / count) ** 2
    var_autocorr = m2_autocorr / count - (mean_autocorr / count) ** 2

    return {
        "mean_power_spectrum": mean_power_spectrum.cpu().numpy(),
        "var_power_spectrum": var_power_spectrum.cpu().numpy(),
        "mean_autocorr": mean_autocorr.cpu().numpy(),
        "var_autocorr": var_autocorr.cpu().numpy(),
    }


def cnn_statistics(
    device: torch.device,
    dataset: Dataset[Tuple[Tensor, int]],
    brain: Brain,
    max_sample_size: int = 0,
) -> Dict[str, Dict[str, FloatArray]]:
    """Compute statistics for a convolutional encoder model.

    This function analyzes the input data and each layer of the convolutional encoder,
    computing various statistical measures and properties.

    Args:
    ----
        device (torch.device): The device to run computations on.
        dataset (Dataset[Tuple[Tensor, int]]): The dataset to analyze.
        encoder (ConvolutionalEncoder): The convolutional encoder model to analyze.
        max_sample_size (int, optional): Maximum number of samples to use. If 0, use all samples. Defaults to 0.

    Returns:
    -------
        Dict[str, Dict[str, FloatArray]]: A nested dictionary containing statistics for the input and each layer.
        The outer dictionary is keyed by layer names (with "input" for the input data), and each inner dictionary
        contains the following keys:

        - "receptive_fields": FloatArray of shape (out_channels, in_channels, height, width)
            Receptive fields for the layer (identity matrix for input).
        - "pixel_histograms": FloatArray of shape (num_channels, num_bins)
            Histograms of pixel values for each channel.
        - "histogram_bin_edges": FloatArray of shape (num_bins + 1,)
            Bin edges for the pixel histograms.
        - "mean_power_spectrum": FloatArray of shape (height, width)
            Mean power spectrum across all channels.
        - "var_power_spectrum": FloatArray of shape (height, width)
            Variance of the power spectrum across all channels.
        - "mean_autocorr": FloatArray of shape (height, width)
            Mean autocorrelation across all channels.
        - "var_autocorr": FloatArray of shape (height, width)
            Variance of the autocorrelation across all channels.
        - "num_channels": FloatArray of shape (1,)
            Number of channels in the layer.

    Note:
    ----
        The function only analyzes activation layers in the encoder's convolutional head.
        If max_sample_size is specified and smaller than the dataset size, a random subset of the data is used.

    """
    input_shape, cnn_layers = get_cnn_circuit(brain)

    # Set sample size
    original_size = len(dataset)
    logger.info(f"Original dataset size: {original_size}")

    if max_sample_size > 0 and original_size > max_sample_size:
        indices: List[int] = torch.randperm(original_size)[:max_sample_size].tolist()
        dataset = Subset(dataset, indices=indices)
        logger.info(f"Reducing dataset size for cnn_statistics to {max_sample_size}")
    else:
        logger.info("Using full dataset for cnn_statistics")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    results: Dict[str, Dict[str, FloatArray]] = {}

    # Compute receptive fields for all layers
    receptive_fields = cnn_linear_receptive_fields(device, input_shape, cnn_layers)

    # Analyze input data
    input_spectral = layer_spectral_analysis(device, dataloader)
    input_histograms = layer_pixel_histograms(device, dataloader)
    # num channels as FloatArray
    num_channels0 = input_shape[0]
    num_channels = np.array(num_channels0, dtype=np.float64)

    results["input"] = {
        "receptive_fields": np.eye(num_channels0)[
            :, :, np.newaxis, np.newaxis
        ],  # Identity for input
        "pixel_histograms": input_histograms["channel_histograms"],
        "histogram_bin_edges": input_histograms["bin_edges"],
        "mean_power_spectrum": input_spectral["mean_power_spectrum"],
        "var_power_spectrum": input_spectral["var_power_spectrum"],
        "mean_autocorr": input_spectral["mean_autocorr"],
        "var_autocorr": input_spectral["var_autocorr"],
        "num_channels": num_channels,
    }

    sublayers: List[nn.Module] = []
    # Analyze each layer
    for name, layer in cnn_layers.items():
        sublayers.append(layer)
        if hasattr(layer, "out_channels"):
            num_channels = layer.out_channels

        # Only analyze input layers
        if is_activation(layer):
            continue

        # Create a temporary model up to the current layer
        temp_model = nn.Sequential(*sublayers)

        # nn.Sequential(
        #             *list(cnn_layers.conv_head.children())[
        #                 : list(cnn_layers.conv_head.named_children()).index((name, layer)) + 1
        #             ]
        #         )

        # Create a new dataloader that applies the temporary model to the data
        transformed_dataloader = _TransformedDataLoader(device, dataloader, temp_model)

        # Perform spectral analysis and histogram computation on the transformed data
        layer_spectral = layer_spectral_analysis(device, transformed_dataloader)
        layer_histograms = layer_pixel_histograms(device, transformed_dataloader)

        results[name] = {
            "receptive_fields": receptive_fields[name],
            "pixel_histograms": layer_histograms["channel_histograms"],
            "histogram_bin_edges": layer_histograms["bin_edges"],
            "mean_power_spectrum": layer_spectral["mean_power_spectrum"],
            "var_power_spectrum": layer_spectral["var_power_spectrum"],
            "mean_autocorr": layer_spectral["mean_autocorr"],
            "var_autocorr": layer_spectral["var_autocorr"],
            "num_channels": np.array([num_channels], dtype=np),
        }

    return results


def get_cnn_circuit(brain: Brain) -> Tuple[Tuple[int, ...], OrderedDict[str, nn.Module]]:
    """Find the longest path starting from a sensor, along a path of ConvolutionalEncoders. This likely won't work very well for particularly complex graphs.

    Returns
    -------
    OrderedDict[str, nn.Module]: A dictionary of layer names and modules.

    """
    cnn_paths: List[List[str]] = []

    # Create for the subgraph of sensors and cnns
    cnn_nodes = [
        node
        for node, circuit in brain.circuits.items()
        if isinstance(circuit, ConvolutionalEncoder)
    ]
    sensor_nodes = [node for node in brain.sensors]
    subgraph = nx.subgraph(brain.connectome, cnn_nodes + sensor_nodes)
    end_nodes = [node for node in cnn_nodes if not list(subgraph.successors(node))]

    for sensor in sensor_nodes:
        cnn_paths.extend(nx.all_simple_paths(subgraph, source=sensor, target=end_nodes))

    # find the longest path
    cnn_path = max(cnn_paths, key=len)
    logger.info(f"Circuit path for analysis: {cnn_path}")
    # Split off the sensor node
    sensor, *cnn_path = cnn_path
    # collect list of cnns
    cnn_circuits = [brain.circuits[node] for node in cnn_path]
    # Combine all cnn layers
    tuples: List[Tuple[str, nn.Module]] = []
    for circuit in cnn_circuits:
        for name, module in circuit.conv_head.named_children():
            tuples.extend([(name, module)])

    input_shape = brain.sensors[sensor]
    return input_shape, OrderedDict(tuples)


class _TransformedDataLoader(DataLoader[Tuple[Tensor, int]]):
    def __init__(
        self,
        device: torch.device,
        dataloader: DataLoader[Tuple[Tensor, int]],
        model: nn.Module,
    ):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.device = device
        super().__init__(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
        )

    def __iter__(self) -> Iterator[Tuple[Tensor, int]]:
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                transformed_inputs = self.model(inputs)
                yield transformed_inputs, labels

    def __len__(self) -> int:
        return len(self.dataloader)
