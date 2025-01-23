from typing import List

import torch
from torch import nn

from retinal_rl.models.neural_circuit import NeuralCircuit


class LatentRNN(NeuralCircuit):
    def __init__(self, input_shape: List[int], rnn_size: int, rnn_num_layers: int):
        super().__init__(input_shape)
        input_size = int(torch.prod(torch.tensor(input_shape)))
        self.core = nn.GRU(input_size, rnn_size, rnn_num_layers)

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)

        if not is_seq:
            head_output = head_output.unsqueeze(0)

        rnn_states = rnn_states.unsqueeze(0)

        # as the last element in x is the new_rnn_states, we don't need to return it
        x, new_rnn_states = self.core(head_output, rnn_states.contiguous())

        if not is_seq:
            x = x.squeeze(0)

        new_rnn_states = new_rnn_states.squeeze(0)

        # x = x.view(-1, *self.output_shape)  # TODO: check how this affects sequence processing
        return x#, new_rnn_states #TODO: Can we just ignore this and extract new_rnn_states from x?


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
