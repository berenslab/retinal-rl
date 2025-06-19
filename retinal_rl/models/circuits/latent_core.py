import torch
from beartype import beartype
from torch import Tensor, nn

from retinal_rl.models.neural_circuit import NeuralCircuit, SimpleNeuralCircuit


class LatentRNN(NeuralCircuit):
    @beartype
    def __init__(self, input_shapes: list[list[int]], rnn_size: int, rnn_num_layers: int):
        super().__init__(input_shapes)
        self.input_size = int(torch.prod(torch.tensor(input_shapes[0])))
        self.core = nn.GRU(self.input_size, rnn_size, rnn_num_layers)

    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Forward pass with tuple interface: (input_data, rnn_states) -> (output, new_rnn_states).
        
        The input_data has shape (batch_size, input_size).
        The rnn_states have shape (batch_size, num_layers * hidden_size).
        Returns (output, new_rnn_states) where output has shape (batch_size, hidden_size)
        and new_rnn_states has shape (batch_size, num_layers * hidden_size).
        """
        input, rnn_states = inputs  # Unpack the two inputs
        is_seq = isinstance(input, torch.nn.utils.rnn.PackedSequence)

        if not is_seq:
            input = input.unsqueeze(0)

        if self.core.num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.core.num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        # as the last element in x is the new_rnn_states, we don't need to return it
        x, new_rnn_states = self.core(input, rnn_states.contiguous())

        if not is_seq:
            x = x.squeeze(0)

        if self.core.num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return (x, new_rnn_states)  # Return as tuple



class LatentFFN(SimpleNeuralCircuit):
    @beartype
    def __init__(self, input_shapes: list[list[int]]):
        super().__init__(input_shapes)

    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        (head_output,) = inputs  # Unpack single input
        # Apply tanh to head output
        output = torch.tanh(head_output)
        return (output,)  # Return as tuple


class LatentIdentity(SimpleNeuralCircuit):
    @beartype
    def __init__(self, input_shapes: list[list[int]]):
        super().__init__(input_shapes)

    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        (head_output,) = inputs  # Unpack single input
        return (head_output,)  # Return as tuple (identity function)
