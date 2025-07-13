import warnings

import torch
from beartype import beartype
from sample_factory.algo.learning.rnn_utils import (
    build_core_out_from_seq,
    build_rnn_inputs,
)
from torch import Tensor, nn

from retinal_rl.models.neural_circuit import NeuralCircuit, SimpleNeuralCircuit


class LatentRNN(NeuralCircuit):
    @beartype
    def __init__(
        self, input_shapes: list[list[int]], rnn_size: int, rnn_num_layers: int
    ):
        super().__init__(input_shapes)
        self.input_size = int(torch.prod(torch.tensor(input_shapes[0])))
        self.core = nn.GRU(self.input_size, rnn_size, rnn_num_layers)

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        input, rnn_states = inputs
        if not self.training or isinstance(rnn_states, torch.Tensor):
            warnings.warn(
                "RNN is in training mode and rnn_states is not a dictionary. Using original forward method."
            )
            output, state = self.core_forward((input, rnn_states))
        else:
            assert all(
                attr in rnn_states
                for attr in ["states", "valids", "dones_cpu", "recurrence"]
            ), "the rnn state must be 'enriched' with valids, dones_cpu, and recurrence attributes when in train mode"

            input_seq, rnn_states, inverted_select_inds = self.prepare_inputs(
                input,
                rnn_states["states"],
                rnn_states["valids"],
                rnn_states["dones_cpu"],
                rnn_states["recurrence"],
            )
            core_output_seq, state = self.core_forward((input_seq, rnn_states))

            output = build_core_out_from_seq(core_output_seq, inverted_select_inds)
        # TODO: Double check state might not be used here or also readjusted
        return (output, state)

    # @beartype remove strict typing for compatibility with samplefactory
    def core_forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
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

    @staticmethod
    def prepare_inputs(inputs, rnn_states, valids, dones_cpu, recurrence):
        # initial rnn states
        # this is the only way to stop RNNs from backpropagating through invalid timesteps
        # (i.e. experience collected by another policy)
        done_or_invalid = torch.logical_or(dones_cpu, ~valids.cpu()).float()
        input_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
            inputs,
            done_or_invalid,
            rnn_states,
            recurrence,
        )
        return input_seq, rnn_states, inverted_select_inds


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
