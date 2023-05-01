"""
retina_rl library

"""
#import torch
from torch import nn

from sample_factory.model.encoder import Encoder
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from sample_factory.algo.utils.context import global_model_factory


### Registration ###


def register_retinal_model(cfg):
    """Registers the retinal model with the global model factory."""
    if cfg.encoder == "retinal":
        log.info("Registering retinal model")
        global_model_factory().register_encoder_factory(make_retinal_encoder)
    elif cfg.encoder == "prototypical":
        log.info("Registering prototypical model")
        global_model_factory().register_encoder_factory(make_prototypical_encoder)
    elif cfg.encoder == "lindsey":
        log.info("Registering lindsey model")
        global_model_factory().register_encoder_factory(make_lindsey_encoder)
    else:
        raise Exception("Unknown model type")


### Model make functions ###


def make_lindsey_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return LindseyEncoder(cfg, obs_space)

def make_prototypical_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return Prototypical(cfg, obs_space)

def make_retinal_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return RetinalEncoder(cfg, obs_space)


### Util ###


def activation(act) -> nn.Module:
    if act == "elu":
        return nn.ELU(inplace=True)
    elif act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity(inplace=True)
    else:
        raise Exception("Unknown activation function")

def is_activation(mdl: nn.Module) -> bool:
    bl = any([isinstance(mdl, nn.ELU)
        ,isinstance(mdl, nn.ReLU)
        ,isinstance(mdl, nn.Tanh)
        ,isinstance(mdl, nn.Identity)])
    return bl

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

        # Number of channels
        self.bipolar_chans = cfg.global_channels
        self.rgc_chans = 2*self.bipolar_chans
        self.bottleneck_chans = cfg.retinal_bottleneck
        self.v1_chans = 2*self.rgc_chans

        # Pooling
        self.spatial_pooling = 2
        self.max_pooling = 4

        # Kernel size
        self.bottleneck_kernel_size = 1
        self.kernel_size = cfg.kernel_size
        self.padding = (self.kernel_size - 1) // 2

        # Preparing Conv Layers
        conv_layers = []

        # bipolar cells
        conv_layers.extend(
                [ nn.Conv2d(3, self.bipolar_chans, self.kernel_size, padding=self.padding)
                    , nn.MaxPool2d(self.spatial_pooling,ceil_mode=True)
                    , activation(self.act_name) ] )
        # ganglion cells
        conv_layers.extend(
                [ nn.Conv2d(self.bipolar_chans, self.rgc_chans, self.kernel_size, padding=self.padding)
                    , nn.MaxPool2d(self.spatial_pooling,ceil_mode=True)
                    , activation(self.act_name) ] )
        # Retinal bottleneck
        conv_layers.extend(
                [ nn.Conv2d(self.rgc_chans, self.bottleneck_chans, self.bottleneck_kernel_size)
                    , activation(self.act_name) ] )
        # V1 Cells
        conv_layers.extend(
                [ nn.Conv2d(self.bottleneck_chans, self.v1_chans, self.kernel_size, padding=self.padding)
                    , nn.MaxPool2d(self.max_pooling,ceil_mode=True)
                    , activation(self.act_name) ] )

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


