import warnings
from typing import List

import torch
from torch import nn

from retinal_rl.models.neural_circuit import NeuralCircuit


class LatentRNN(NeuralCircuit):
    def __init__(self, input_shape: List[int], rnn_size: int, rnn_num_layers: int):
        super().__init__(input_shape)
        self.input_size = int(torch.prod(torch.tensor(input_shape)))
        self.core = nn.GRU(self.input_size, rnn_size, rnn_num_layers)

    def forward(
        self,
        input: torch.Tensor | torch.nn.utils.rnn.PackedSequence,
        rnn_states: torch.Tensor,
    ):
        """
        Does some reshaping work so that tensor and packed sequences can be processed.
        Everything is expected to be batched, so the input has shape (batch_size, input_size).
        The rnn_states are expected to be of shape (batch_size, num_layers * hidden_size).
        On the other hand, the output is of shape (batch_size, hidden_size), and the new_rnn_states are of shape (batch_size, num_layers * hidden_size).
        """
        # TODO: find better way / abstraction for NeuralCircuits with more complex function signatures
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

        return x, new_rnn_states

    @NeuralCircuit.input_shape.getter
    def input_shape(self) -> list[list[int]]:
        warnings.warn(
            "LatentRNN input and output shapes might not be compatible with other circuits.",
            UserWarning,
        )
        return [
            [1, self.input_size],
            [1, self.core.num_layers * self.core.hidden_size],
        ]  # LatentRNN can only handle batched inputs

    @NeuralCircuit.output_shape.getter
    def output_shape(self) -> list[list[int]]:
        warnings.warn(
            "LatentRNN input and output shapes might not be compatible with other circuits.",
            UserWarning,
        )
        with torch.no_grad():
            return [
                [1, self.core.hidden_size],
                [1, self.core.num_layers * self.core.hidden_size],
            ]


class LatentFFN(NeuralCircuit):
    def __init__(self, input_shape: List[int]):
        super().__init__(input_shape)
        self.core_output_size = input_shape

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states=None):
        # Apply tanh to head output
        return torch.tanh(head_output)  # , fake_rnn_states

    @NeuralCircuit.output_shape.getter
    def output_shape(self) -> List[int]:  # TODO: fake_rnn_states?
        return self.core_output_size


class LatentIdentity(NeuralCircuit):
    def __init__(self, input_shape: List[int]):
        super().__init__(input_shape)
        self.core_output_size = input_shape

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states=None):
        return head_output  # , fake_rnn_states

    @NeuralCircuit.output_shape.getter
    def output_shape(self) -> List[int]:
        return self.core_output_size
