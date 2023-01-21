"""
retina_rl library

"""
import torch
from torch import nn
from torchvision.transforms import Grayscale

from sample_factory.model.encoder import Encoder
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from sample_factory.algo.utils.context import global_model_factory


### Registration ###


def register_retinal_model():
    global_model_factory().register_encoder_factory(make_lindsey_encoder)


### Retinal-VVS Model ###


def make_lindsey_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return LindseyEncoder(cfg, obs_space)


def activation(cfg: Config) -> nn.Module:
    if cfg.activation == "elu":
        return nn.ELU(inplace=True)
    elif cfg.activation == "relu":
        return nn.ReLU(inplace=True)
    elif cfg.activation == "tanh":
        return nn.Tanh()
    elif cfg.activation == "linear":
        return nn.Identity(inplace=True)
    else:
        raise Exception("Unknown activation function")


class LindseyEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        # reuse the default image encoder
        #self.basic_encoder = LindseyEncoderBase(cfg, obs_space)

        self.basic_encoder = LindseyEncoderBase(cfg, obs_space["obs"])
        self.encoder_out_size = self.basic_encoder.get_out_size()
        self.gs = cfg.greyscale

        self.measurements_head = None
        if "measurements" in obs_space.keys():
            self.measurements_head = nn.Sequential(
                nn.Linear(obs_space["measurements"].shape[0], 128),
                activation(cfg),
                nn.Linear(128, 128),
                activation(cfg),
            )
            measurements_out_size = calc_num_elements(self.measurements_head, obs_space["measurements"].shape)
            self.encoder_out_size += measurements_out_size

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict["obs"])

        if self.gs:
            gs = Grayscale(num_output_channels=1) # change compared to vanilla Lindsey
            x = gs.forward(x) # change compared to vanilla Lindsey


        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict["measurements"].float())
            x = torch.cat((x, measurements), dim=1)

        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

class LindseyEncoderBase(Encoder):

    def __init__(self, cfg : Config , obs_space : ObsSpace):

        super().__init__(cfg)

        nchns = cfg.global_channels
        btlchns = cfg.retinal_bottleneck
        vvsdpth = cfg.vvs_depth
        krnsz = cfg.kernel_size
        retstrd = cfg.retinal_stride # only for first conv layer

        self.nl_fc = activation(cfg)

        self.kernel_size = krnsz

        # Preparing Conv Layers
        conv_layers = []
        self.nls = []
        for i in range(vvsdpth+2): # +2 for the first 'retinal' layers

            self.nls.append(activation(cfg))

            if i == 0: # 'bipolar cells' ('global channels')
                conv_layers.extend([nn.Conv2d(3, nchns, krnsz, stride=retstrd), self.nls[i]])
            elif i == 1: # 'ganglion cells' ('retinal bottleneck')
                conv_layers.extend([nn.Conv2d(nchns, btlchns, krnsz, stride=1), self.nls[i]])
            elif i == 2: # 'V1' ('global channels')
                conv_layers.extend([nn.Conv2d(btlchns, nchns, krnsz, stride=1), self.nls[i]])
            else: # 'vvs layers'
                conv_layers.extend([nn.Conv2d(nchns, nchns, krnsz, stride=1), self.nls[i]])

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space.shape)
        self.encoder_out_size = cfg.rnn_size
        self.fc1 = nn.Linear(self.conv_head_out_size,self.encoder_out_size)

    def forward(self, x):

        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size
