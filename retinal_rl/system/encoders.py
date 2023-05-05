"""
retina_rl library

"""
#import torch
from torch import nn
from collections import OrderedDict

from sample_factory.model.encoder import Encoder
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from sample_factory.algo.utils.context import global_model_factory

from retinal_rl.util import activation


### Registration ###


def register_retinal_model():
    """Registers the retinal model with the global model factory."""
    global_model_factory().register_encoder_factory(make_encoder)

### Model make functions ###


def make_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """defines the encoder constructor."""
    if cfg.encoder == "retinal":
        return RetinalEncoder(cfg, obs_space)
    elif cfg.encoder == "prototypical":
        return Prototypical(cfg, obs_space)
    elif cfg.encoder == "lindsey":
        return LindseyEncoder(cfg, obs_space)
    else:
        raise Exception("Unknown model type")




### Retinal Encoder ###


class RetinalEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        #self.basic_encoder = torch.jit.script(LindseyEncoderBase(cfg, obs_space["obs"]))
        self.basic_encoder = RetinalEncoderBase(cfg, obs_space["obs"])

        self.encoder_out_size = self.basic_encoder.get_out_size()

        log.debug("Policy head output size: %r", self.get_out_size())
        self.encoder_out_size = self.basic_encoder.get_out_size()

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict["obs"])
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

class RetinalEncoderBase(Encoder):

    def __init__(self, cfg : Config , obs_space : ObsSpace):

        super().__init__(cfg)

        # Activation function
        self.act_name = cfg.activation
        self.nl_fc = activation(cfg.activation)

        # Saving parameters
        self.gchans = cfg.global_channels
        self.bchans = cfg.retinal_bottleneck
        self.krnsz = cfg.kernel_size

        # Pooling
        self.spool = 2
        self.mpool = 4

        # Padding
        self.cpad = (self.krnsz - 1) // 2
        self.spad = (self.spool - 1) // 2
        self.mpad = (self.mpool - 1) // 2

        # Preparing Conv Layers
        conv_layers = OrderedDict(

                [ ('bp_filters', nn.Conv2d(3, self.gchans, self.krnsz, padding=self.cpad))
                , ('bp_outputs', activation(self.act_name))
                , ('bp_averages', nn.AvgPool2d(self.spool,padding=self.spad))

                , ('rgc_filters', nn.Conv2d(self.gchans, self.gchans, self.krnsz, padding=self.cpad))
                , ('rgc_outputs', activation(self.act_name))
                , ('rgc_averages', nn.AvgPool2d(self.spool,padding=self.spad))

                , ('lgn_filters', nn.Conv2d(self.gchans, self.bchans, self.krnsz, self.cpad))
                , ('lgn_outputs', activation(self.act_name))
                , ('lgn_averages', nn.AvgPool2d(self.spool,padding=self.spad))

                , ('v1_filters', nn.Conv2d(self.bchans, self.gchans, self.krnsz, padding=self.cpad))
                , ('simple_outputs', activation(self.act_name))
                , ('complex_outputs', nn.MaxPool2d(self.mpool,padding=self.mpad))

                ] )

        self.conv_head = nn.Sequential(conv_layers)
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


### Retinal-VVS Model ###


class LindseyEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        #self.basic_encoder = torch.jit.script(LindseyEncoderBase(cfg, obs_space["obs"]))
        self.basic_encoder = LindseyEncoderBase(cfg, obs_space["obs"])

        self.encoder_out_size = self.basic_encoder.get_out_size()

        log.debug("Policy head output size: %r", self.get_out_size())
        self.encoder_out_size = self.basic_encoder.get_out_size()

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict["obs"])
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

            self.nls.append(activation(cfg.activation))

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


### Lindsey based encoder with max pooling for testing ###


class PrototypicalBase(Encoder):

    def __init__(self, cfg : Config , obs_space : ObsSpace):

        super().__init__(cfg)

        self.nl_fc = activation(cfg)
        self.act_name = cfg.activation
        self.kernel_size = 5

        # Preparing Conv Layers
        conv_layers = []

        self.padding = (self.kernel_size - 1) // 2

        conv_layers.extend(
                [nn.Conv2d(3, 16, self.kernel_size,padding=self.padding)
                    , nn.MaxPool2d(2)
                    , activation(self.act_name) ])
        conv_layers.extend(
                [nn.Conv2d(16, 16, self.kernel_size,padding=self.padding)
                    , activation(self.act_name) ])
        conv_layers.extend(
                [nn.Conv2d(16, 8, self.kernel_size,padding=self.padding)
                    , nn.MaxPool2d(2)
                    , activation(self.act_name) ])

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


class Prototypical(Encoder):

    def __init__(self, cfg: Config, obs_space: ObsSpace):

        super().__init__(cfg)

        #self.basic_encoder = torch.jit.script(LindseyEncoderBaseMaxPool(cfg, obs_space["obs"]))
        self.basic_encoder = PrototypicalBase(cfg, obs_space["obs"])

        self.encoder_out_size = self.basic_encoder.get_out_size()

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        x = self.basic_encoder(obs_dict["obs"])
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


