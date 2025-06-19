"""Linear classifier neural circuit."""

import torch
from beartype import beartype
from torch import Tensor

from retinal_rl.models.neural_circuit import SimpleNeuralCircuit


class LinearClassifier(SimpleNeuralCircuit):
    @beartype
    def __init__(
        self,
        input_shapes: list[list[int]],
        num_classes: int,
    ):
        super().__init__(input_shapes)

        input_size = int(torch.prod(torch.tensor(self.input_shape)))
        self.fc = torch.nn.Linear(input_size, num_classes)

    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        (x,) = inputs  # Unpack single input
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        if not self.training:
            x = torch.nn.functional.softmax(x, dim=1)
        return (x,)  # Return as tuple
