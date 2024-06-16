from collections import OrderedDict

import torch
import torch.nn as nn

from retinal_rl_models.base_model import BaseModel
from retinal_rl_models.util import assert_list


class GenericModel(BaseModel):
    def __init__(
        self,
        inp_shape: tuple[int, int, int],
        out_size: int,
        num_layers: int = 3,
        fc_in_size: int = None,
        num_channels: int | list[int] = 16,
        kernel_size: int | list[int] = 3,
        stride: int | list[int] = 1,
        padding: int | list[int] = 0,
        dilation: int | list[int] = 1,
        act_name="relu",
        pooling_ks: int | list[int] = 1,
    ):
        # add parameters to model and apply changes for internal use
        super().__init__(locals())
        self.padding = assert_list(padding, self.num_layers)
        self.dilation = assert_list(dilation, self.num_layers)
        self.num_channels = assert_list(num_channels, self.num_layers)
        self.kernel_size = assert_list(kernel_size, self.num_layers)
        self.stride = assert_list(stride, self.num_layers)
        self.pooling_ks = assert_list(pooling_ks, self.num_layers)

        conv_layers = []
        # Define convolutional layers
        for i in range(num_layers):
            in_channels = self.inp_shape[0] if i == 0 else self.num_channels[i - 1]
            conv_layers.append(
                (
                    "conv" + str(i),
                    torch.nn.Conv2d(
                        in_channels,
                        self.num_channels[i],
                        self.kernel_size[i],
                        self.stride[i],
                        self.padding[i],
                        self.dilation[i],
                    ),
                )
            )
            conv_layers.append(
                (self.act_name + str(i), self.str_to_activation(self.act_name))
            )
            if i == num_layers - 1 and self.fc_in_size != None:
                conv_layers.append(
                    ("pool" + str(i), nn.AdaptiveAvgPool2d(output_size=fc_in_size))
                )
            elif self.pooling_ks[i] > 1:
                conv_layers.append(
                    ("pool" + str(i), nn.AvgPool2d(kernel_size=self.pooling_ks[i]))
                )
        self.conv_head = nn.Sequential(OrderedDict(conv_layers))

        # Fully connected layer
        fc_in = self.calc_num_elements(self.conv_head, self.inp_shape)
        self.fc = nn.Linear(fc_in, self.out_size)
        self.nl_fc = self.str_to_activation(self.act_name)

    def forward(self, x):
        x = self.conv_head(x)
        x = x.view(x.size(0), -1)
        x = self.nl_fc(self.fc(x))
        return x
