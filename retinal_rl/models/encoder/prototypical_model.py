from collections import OrderedDict

from torch import nn

from retinal_rl_models.base_model import BaseModel


# Prototypical Encoder
class PrototypicalModel(BaseModel):
    def __init__(
        self, out_size: int, inp_shape: tuple[int, int], act_name: str = "ELU"
    ):
        super().__init__(locals())

        self.act_name = act_name
        self.encoder_out_size = out_size
        self.nl_fc = self.str_to_activation(act_name)
        conv_layers = OrderedDict(
            [
                ("conv1_filters", nn.Conv2d(3, 32, 8, stride=4)),
                ("conv1_output", self.str_to_activation(self.act_name)),
                ("conv2_filters", nn.Conv2d(32, 64, 4, stride=2)),
                ("conv2_output", self.str_to_activation(self.act_name)),
                ("conv3_filters", nn.Conv2d(64, 128, 3, stride=2)),
                ("conv3_output", self.str_to_activation(self.act_name)),
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
