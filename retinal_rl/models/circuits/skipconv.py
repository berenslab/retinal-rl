from typing import List, Optional, Union

import torch
from torchvision.transforms.functional import center_crop

from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder
from retinal_rl.models.neural_circuit import NeuralCircuit


class SkipConvolution(NeuralCircuit):
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
        add: bool = False,
    ):
        self.conv = ConvolutionalEncoder(
            input_shape,
            num_layers,
            num_channels,
            kernel_size,
            stride,
            activation,
            layer_norm,
            affine_norm,
            layer_names,
        )
        self.add = add

        if self.add:
            last_channels = (
                num_channels if isinstance(num_channels, int) else num_channels[-1]
            )
            assert (
                last_channels == input_shape[0]
            ), "If the skip connection should be added to the output (ResNet style), the number of input and output channels must match."

    def forward(self, x: torch.Tensor):
        output = self.conv(x)

        # Crop input
        cropped_input = center_crop(x, output.shape)

        if self.add:
            output = output + cropped_input
        else:
            output = torch.concat([output, cropped_input], dim=1)

        return output
