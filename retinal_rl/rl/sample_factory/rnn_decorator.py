import warnings
from functools import wraps
from inspect import signature

import torch
from sample_factory.algo.learning.rnn_utils import (
    build_core_out_from_seq,
    build_rnn_inputs,
)

from retinal_rl.models.circuits.latent_core import LatentRNN


def decorate_forward(rnn: LatentRNN):
    """
    Decorator to wrap the forward method of an RNNBase object.
    This decorator is used to modify the forward method of an RNNBase object to include the necessary pre-processing
    steps before calling the original forward method.

    Args:
        rnn: The object for which the forward function should be modified.

    Returns:
        The modified forward method of the RNNBase object.

    Usage:
    ```
    rnn = nn.LSTM(10, 20, 2)
    decorate_forward(rnn)
    ```
    """
    orig_method = rnn.forward

    @wraps(orig_method)
    def wrapper(input, rnn_states):
        if not rnn.training or isinstance(rnn_states, torch.Tensor):
            warnings.warn(
                "RNN is in training mode and rnn_states is not a dictionary. Using original forward method."
            )
            output = orig_method(input, rnn_states)
        else:
            assert all(
                attr in rnn_states
                for attr in ["states", "valids", "dones_cpu", "recurrence"]
            ), "the rnn state must be 'enriched' with valids, dones_cpu, and recurrence attributes when in train mode"

            input_seq, rnn_states, inverted_select_inds = prepare_inputs(
                input,
                rnn_states["states"],
                rnn_states["valids"],
                rnn_states["dones_cpu"],
                rnn_states["recurrence"],
            )
            core_output_seq = orig_method(input_seq, rnn_states)

            output = build_core_out_from_seq(core_output_seq, inverted_select_inds)
        return output  # , state

    setattr(rnn, orig_method.__name__, wrapper)
    return wrapper


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
