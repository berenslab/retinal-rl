"""Contains the implementation of convolutional encoder and decoder neural circuits."""

import logging
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from retinal_rl.models.neural_circuit import NeuralCircuit
from retinal_rl.util import assert_list

logger = logging.getLogger(__name__)


def _calculate_padding(kernel_size: int, stride: int) -> int:
    div, mod = divmod(kernel_size - stride, 2)
    if mod != 0:
        logger.error(
            f"Invalid kernel size {kernel_size} and stride {stride}, KS - S must be divisible by 2."
        )
        raise ValueError("Invalid kernel size and stride combination.")
    return div


class ConvolutionalEncoder(NeuralCircuit):
    """A convolutional encoder that applies a series of convolutional layers to input data.

    Args:
    ----
        input_shape (List[int]): The shape of the input tensor (e.g., [channels, height, width]).
        num_layers (int): The number of convolutional layers. Default is 3.
        num_channels (List[int]): The number of channels for each layer. Default is 16.
        kernel_size (Union[int, List[int]]): The size of the convolutional kernels. Default is 3.
        stride (Union[int, List[int]]): The stride for the convolutional layers. Default is 1.
        activation (str): The name of the activation function to use. Default is "relu".

    """

    def __init__(
        self,
        input_shape: List[int],
        num_layers: int,
        num_channels: Union[int, List[int]],
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        activation: Union[str, List[str]],
        layer_norm: bool = False,
        affine_norm: bool = True,
        layer_names: Optional[List[str]] = None,
    ):
        super().__init__(input_shape)
        self.num_layers = num_layers
        self.num_channels = assert_list(num_channels, self.num_layers)
        self.kernel_size = assert_list(kernel_size, self.num_layers)
        self.stride = assert_list(stride, self.num_layers)
        self.activation = assert_list(activation, self.num_layers)

        self.padding: List[int] = []
        for i in range(num_layers):
            self.padding.append(_calculate_padding(self.kernel_size[i], self.stride[i]))
        conv_layers: List[Tuple[str, nn.Module]] = []
        # Define convolutional layers
        for i in range(num_layers):
            in_channels = self.input_shape[0] if i == 0 else self.num_channels[i - 1]
            lyrnm = (
                f"{layer_names[i]}_input"
                if layer_names is not None
                else "conv" + str(i)
            )
            actnm = (
                f"{layer_names[i]}_activation"
                if layer_names is not None
                else self.activation[i] + str(i)
            )
            conv_layers.append(
                (
                    lyrnm,
                    torch.nn.Conv2d(
                        in_channels,
                        self.num_channels[i],
                        self.kernel_size[i],
                        self.stride[i],
                        self.padding[i],
                    ),
                )
            )
            if layer_norm and i is num_layers - 1:
                conv_layers.append(
                    (
                        f"{layer_names[i]}_instance_norm"
                        if layer_names is not None
                        else "instance_norm" + str(i),
                        nn.InstanceNorm2d(self.num_channels[i], affine=affine_norm),
                    )
                )
            conv_layers.append((actnm, self.str_to_activation(self.activation[i])))
        self.conv_head = nn.Sequential(OrderedDict(conv_layers))

    def forward(self, x: Tensor):
        return self.conv_head(x)


class ConvolutionalDecoder(NeuralCircuit):
    """A convolutional decoder that applies a series of deconvolutional layers to reconstruct data from encoded input."""

    def __init__(
        self,
        input_shape: List[int],
        num_layers: int,
        num_channels: Union[int, List[int]],
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        activation: Union[str, List[str]],
        layer_norm: bool = False,
        affine_norm: bool = True,
        layer_names: Optional[List[str]] = None,
    ):
        # add parameters to model and apply changes for internal use
        super().__init__(input_shape)

        self.num_layers = num_layers
        self.num_channels = assert_list(num_channels, self.num_layers)
        self.kernel_size = assert_list(kernel_size, self.num_layers)
        self.stride = assert_list(stride, self.num_layers)
        self.activation = assert_list(activation, self.num_layers)

        self.padding: List[int] = []
        for i in range(num_layers):
            self.padding.append(_calculate_padding(self.kernel_size[i], self.stride[i]))

        deconv_layers: List[Tuple[str, nn.Module]] = []
        # Define deconvolutional layers
        for i in range(num_layers):
            in_channels = self.input_shape[0] if i == 0 else self.num_channels[i - 1]
            lyrnm = (
                f"{layer_names[i]}_input"
                if layer_names is not None
                else "deconv" + str(i)
            )
            actnm = (
                f"{layer_names[i]}_activation"
                if layer_names is not None
                else self.activation[i] + str(i)
            )

            deconv_layers.append(
                (
                    lyrnm,
                    torch.nn.ConvTranspose2d(
                        in_channels,
                        self.num_channels[i],
                        self.kernel_size[i],
                        self.stride[i],
                        self.padding[i],
                    ),
                )
            )
            if layer_norm and i == 0:
                deconv_layers.append(
                    (
                        f"{layer_names[i]}_instance_norm"
                        if layer_names is not None
                        else "instance_norm" + str(i),
                        nn.InstanceNorm2d(self.num_channels[i], affine=affine_norm),
                    )
                )

            deconv_layers.append((actnm, self.str_to_activation(self.activation[i])))
        self.deconv_head = nn.Sequential(OrderedDict(deconv_layers))

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv_head(x)
