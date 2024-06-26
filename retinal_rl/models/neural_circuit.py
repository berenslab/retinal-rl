from abc import ABC

import torch
import torch.nn as nn
import torchscan

from typing import Tuple


class NeuralCircuit(nn.Module, ABC):
    def __init__(self) -> None:
        """
        Initializes the base model.
        All params in the dictionary will be added as instance parameters.

        parameters: the parameters used to instantiate a model. Simplest way to pass them on: call locals()
        """
        super().__init__()

    def scan(self, input_size: Tuple[int]) -> None:
        """
        Runs torchscan on the model.

        Args:
            input_size (tuple): Size of the input tensor (batch_size, channels, height, width).
        """
        torchscan.summary(self, input_size, receptive_field=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x
        raise NotImplementedError(
            "Each subclass must implement its own forward method."
        )

    @staticmethod
    def str_to_activation(act: str) -> nn.Module:
        act = str.lower(act)
        if act == "elu":
            return nn.ELU(inplace=True)
        elif act == "relu":
            return nn.ReLU(inplace=True)
        elif act == "tanh":
            return nn.Tanh()
        elif act == "softplus":
            return nn.Softplus()
        elif act == "identity":
            return nn.Identity(inplace=True)
        else:
            raise Exception("Unknown activation function")

    @staticmethod
    def calc_num_elements(module: nn.Module, module_input_shape: list[int]):
        shape_with_batch_dim = (1,) + tuple(module_input_shape)
        some_input = torch.rand(shape_with_batch_dim)
        num_elements = module(some_input).numel()
        return num_elements
