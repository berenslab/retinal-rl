"""Fully connected neural circuits for encoding and decoding data."""

from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from retinal_rl.models.neural_circuit import NeuralCircuit


class FullyConnected(NeuralCircuit):
    """A fully connected layer that applies a series of fully connected layers to input data.

    Args:
    ----
        input_shape (List[int]): The shape of the input tensor (e.g., [channels, height, width]).
        output_shape (List[int]): The shape of the output tensor.
        hidden_units (Union[int, List[int]]): The number of hidden units for each layer. Default is 128.
        activation (str): The name of the activation function to use. Default is "relu".

    """

    def __init__(
        self,
        input_shape: List[int],
        output_shape: List[int],
        activation: Optional[str],
        hidden_units: List[int] = [],
    ):
        super().__init__(input_shape)

        self._output_shape = output_shape
        self.hidden_units = hidden_units
        self.activation = activation

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
            if self.activation is not None:
                fc_layers.append(
                    (self.activation + str(i), self.str_to_activation(self.activation))
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
