from typing import OrderedDict

import torch
from torch.nn import AvgPool2d, Conv2d, MaxPool2d, Sequential

from retinal_rl.models.neural_circuit import SimpleNeuralCircuit


class RetinalEncoder(SimpleNeuralCircuit):
    """TODO: Description"""

    def __init__(
        self, input_shapes: list[list[int]], bp_channels: int, act_name: str
    ):
        super().__init__(input_shapes)

        # Activation function
        self.act_name = act_name

        # Saving parameters
        self.bp_channels = bp_channels
        self.rgc_chans = self.bp_channels * 2
        self.v1_chans = self.rgc_chans * 2

        btl_chans = None
        if btl_chans is not None:
            self.btl_chans = btl_chans
        else:
            self.btl_chans = self.rgc_chans

        # Pooling
        spool = 3
        mpool = 4

        # Padding
        pad = 0

        # Preparing Conv Layers
        layers = [
            (
                "bp_filters",
                Conv2d(3, self.bp_channels, spool, padding=pad),
            ),
            ("bp_outputs", self.str_to_activation(self.act_name)),
            ("bp_averages", AvgPool2d(spool, ceil_mode=True)),
            (
                "rgc_filters",
                Conv2d(self.bp_channels, self.rgc_chans, spool, padding=pad),
            ),
            ("rgc_outputs", self.str_to_activation(self.act_name)),
            ("rgc_averages", AvgPool2d(spool, ceil_mode=True)),
            ("btl_filters", Conv2d(self.rgc_chans, self.btl_chans, 1)),
            ("btl_outputs", self.str_to_activation(self.act_name)),
            (
                "v1_filters",
                Conv2d(self.btl_chans, self.v1_chans, mpool, padding=pad),
            ),
            ("v1_simple_outputs", self.str_to_activation(self.act_name)),
            ("v1_complex_outputs", MaxPool2d(mpool, ceil_mode=True)),
        ]

        self.conv_head = Sequential(OrderedDict(layers))

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return (self.conv_head(inputs[0]),)
