"""Fully connected neural circuits for encoding and decoding data."""

from collections import OrderedDict

import torch
from beartype import beartype
from torch import Tensor, nn

from retinal_rl.models.neural_circuit import SimpleNeuralCircuit


class FullyConnected(SimpleNeuralCircuit):
    """A fully connected layer that applies a series of fully connected layers to input data.

    Args:
    ----
        input_shape (list[int]): The shape of the input tensor (e.g., [channels, height, width]).
        output_shape (list[int]): The shape of the output tensor.
        hidden_units (Union[int, list[int]]): The number of hidden units for each layer. Default is 128.
        activation (str): The name of the activation function to use. Default is "relu".

    """

    @beartype
    def __init__(
        self,
        input_shapes: list[list[int]],
        output_shape: list[int],
        activation: str | None,
        hidden_units: list[int] = [],
    ):
        super().__init__(input_shapes)

        self._output_shape = tuple(output_shape)
        self.hidden_units = hidden_units
        self.activation = activation

        num_layers = len(hidden_units) + 1

        fc_layers: list[tuple[str, nn.Module]] = []
        input_size = int(torch.prod(torch.tensor(self.input_shape)))
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


    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        (x,) = inputs  # Unpack single input
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc_head(x)
        output = x.view(-1, *self._output_shape)  # Reshape to the output shape
        return (output,)  # Return as tuple
