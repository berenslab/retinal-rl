import torch
from beartype import beartype
from torch import Tensor, nn

from retinal_rl.models.neural_circuit import SimpleNeuralCircuit


class Actor(SimpleNeuralCircuit):
    """
    A single fully connected layer that maps from its input to the number of actions.
    This implements the "actor" part of an actor-critic architecture and needs to be present in RL configurations.
    """

    @beartype
    def __init__(
        self,
        input_shapes: list[list[int]],
        num_actions: int,
    ):
        super().__init__(input_shapes)
        input_size = int(torch.prod(torch.tensor(self.input_shape)))
        self.fc = nn.Linear(input_size, num_actions)

    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        (x,) = inputs  # Unpack single input
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return (x,)  # Return as tuple


class Critic(SimpleNeuralCircuit):
    """
    A single fully connected layer that maps from its input to a single scalar value.
    This implements the "critic" part of an actor-critic architecture and needs to be present in RL configurations.
    """

    @beartype
    def __init__(
        self,
        input_shapes: list[list[int]],
    ):
        super().__init__(input_shapes)
        input_size = int(torch.prod(torch.tensor(self.input_shape)))
        self.fc = nn.Linear(input_size, 1)

    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        (x,) = inputs  # Unpack single input
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return (x,)  # Return as tuple
