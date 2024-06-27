"""Linear classifier neural circuit."""

from typing import List

import torch

from retinal_rl.models.neural_circuit import NeuralCircuit


class LinearClassifier(NeuralCircuit):
    def __init__(
        self,
        input_shape: List[int],
        num_classes: int,
        act_name: str,
    ):
        super().__init__(input_shape)

        self.act_name = act_name
        input_size = int(torch.prod(torch.tensor(input_shape)))
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        if not self.training:
            x = torch.nn.functional.softmax(x, dim=1)
        return x
