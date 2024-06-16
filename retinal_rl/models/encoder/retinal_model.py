from collections import OrderedDict

from torch import nn

from retinal_rl_models.base_model import BaseModel


class RetinalModel(BaseModel):
    def __init__(
        self,
        base_channels: int,
        out_size: int,
        inp_shape: tuple[int, int, int],
        retinal_bottleneck: int = None,
        act_name: str = "ELU",
    ):
        super().__init__(locals())

        self.act_name = act_name
        self.encoder_out_size = out_size

        # Activation function
        self.nl_fc = self.str_to_activation(self.act_name)

        # Saving parameters
        self.bp_chans = base_channels
        self.rgc_chans = self.bp_chans * 2
        self.v1_chans = self.rgc_chans * 2

        if retinal_bottleneck is not None:
            self.btl_chans = retinal_bottleneck
        else:
            self.btl_chans = self.rgc_chans

        # Pooling
        self.spool = 3
        self.mpool = 4

        # Padding
        self.spad = 0  # padder(self.spool)
        self.mpad = 0  # padder(self.mpool)

        # Preparing Conv Layers
        conv_layers = OrderedDict(
            [
                (
                    "bp_filters",
                    nn.Conv2d(3, self.bp_chans, self.spool, padding=self.spad),
                ),
                ("bp_outputs", self.str_to_activation(self.act_name)),
                ("bp_averages", nn.AvgPool2d(self.spool, ceil_mode=True)),
                (
                    "rgc_filters",
                    nn.Conv2d(
                        self.bp_chans, self.rgc_chans, self.spool, padding=self.spad
                    ),
                ),
                ("rgc_outputs", self.str_to_activation(self.act_name)),
                ("rgc_averages", nn.AvgPool2d(self.spool, ceil_mode=True)),
                ("btl_filters", nn.Conv2d(self.rgc_chans, self.btl_chans, 1)),
                ("btl_outputs", self.str_to_activation(self.act_name)),
                (
                    "v1_filters",
                    nn.Conv2d(
                        self.btl_chans, self.v1_chans, self.mpool, padding=self.mpad
                    ),
                ),
                ("v1_simple_outputs", self.str_to_activation(self.act_name)),
                ("v1_complex_outputs", nn.MaxPool2d(self.mpool, ceil_mode=True)),
            ]
        )

        self.conv_head = nn.Sequential(conv_layers)

        self.conv_head_out_size = self.calc_num_elements(self.conv_head, inp_shape)
        self.fc1 = nn.Linear(self.conv_head_out_size, self.encoder_out_size)

    def forward(self, x):
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size
