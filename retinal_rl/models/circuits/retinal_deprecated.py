from typing import List, OrderedDict

import torch
from torch.nn import AvgPool2d, Conv2d, Flatten, Linear, MaxPool2d, Sequential

from retinal_rl.models.neural_circuit import NeuralCircuit


class RetinalEncoder(NeuralCircuit):
    """TODO: Description"""

    def __init__(
        self, input_shape: List[int], bp_channels: int, act_name: str, out_shape: int
    ):
        super().__init__(input_shape)

        # Activation function
        self.act_name = act_name
        self.nl_fc = self.str_to_activation(self.act_name)

        # Saving parameters
        self.bp_channels = bp_channels
        self.rgc_chans = self.bp_channels * 2
        self.v1_chans = self.rgc_chans * 2
        self.out_shpe = out_shape

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
        self.flatten = Flatten()
        _test_inp = torch.empty(1, *self.input_shape)
        _test_out = self.flatten(self.conv_head(_test_inp))
        self.fc = Linear(_test_out.shape[1], out_shape)

    def forward(self, x: torch.Tensor):
        x = self.flatten(self.conv_head(x))
        x = self.nl_fc(self.fc(x))
        return x
