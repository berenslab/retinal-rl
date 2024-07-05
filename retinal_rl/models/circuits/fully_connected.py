"""Fully connected neural circuits for encoding and decoding data."""

from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from retinal_rl.models.neural_circuit import NeuralCircuit


class FullyConnectedEncoder(NeuralCircuit):
    """A fully connected encoder that applies a series of fully connected layers to input data.

    Args:
    ----
        input_shape (List[int]): The shape of the input tensor (e.g., [channels, height, width]).
        output_shape (List[int]): The shape of the output tensor.
        hidden_units (Union[int, List[int]]): The number of hidden units for each layer. Default is 128.
        act_name (str): The name of the activation function to use. Default is "relu".

    """

    def __init__(
        self,
        input_shape: List[int],
        output_shape: List[int],
        hidden_units: List[int],
        act_name: str,
    ):
        super().__init__(input_shape)

        self._output_shape = output_shape
        self.hidden_units = hidden_units
        self.act_name = act_name

        num_layers = len(hidden_units) + 1

        fc_layers: List[Tuple[str, nn.Module]] = []
        input_size = int(torch.prod(torch.tensor(input_shape)))
        for i in range(num_layers):
            output_size = (
                self.hidden_units[i]
                if i < num_layers - 1
                else int(torch.prod(torch.tensor(output_shape)))
            )
            fc_layers.append(
                (
                    "fc" + str(i),
                    nn.Linear(input_size, output_size),
                )
            )
            fc_layers.append(
                (self.act_name + str(i), self.str_to_activation(self.act_name))
            )
            input_size = output_size

        self.fc_head = nn.Sequential(OrderedDict(fc_layers))

    @property
    def output_shape(self) -> List[int]:
        """Return the shape of the output tensor."""
        return self._output_shape

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc_head(x)
        return x.view(-1, *self.output_shape)  # Reshape to the output shape


class FullyConnectedDecoder(NeuralCircuit):
    """A fully connected decoder that applies a series of fully connected layers to reconstruct data from encoded input."""

    def __init__(
        self,
        input_shape: List[int],
        output_shape: List[int],
        hidden_units: List[int],
        act_name: str,
    ):
        super().__init__(input_shape)

        self._output_shape = output_shape
        self.hidden_units = hidden_units
        self.act_name = act_name
        num_layers = len(hidden_units) + 1

        fc_layers: List[Tuple[str, nn.Module]] = []
        input_size = int(torch.prod(torch.tensor(input_shape)))
        for i in range(num_layers):
            output_size = (
                self.hidden_units[i]
                if i < num_layers - 1
                else int(torch.prod(torch.tensor(output_shape)))
            )
            fc_layers.append(
                (
                    "fc" + str(i),
                    nn.Linear(input_size, output_size),
                )
            )
            fc_layers.append(
                (self.act_name + str(i), self.str_to_activation(self.act_name))
            )
            input_size = output_size

        self.fc_head = nn.Sequential(OrderedDict(fc_layers))

    @property
    def output_shape(self) -> List[int]:
        """Return the shape of the output tensor."""
        return self._output_shape

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc_head(x)
        return x.view(-1, *self.output_shape)  # Reshape to the output shape
