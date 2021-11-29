"""
retina_rl library

"""

import sys

import numpy as np
from torch import nn

from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape, nonlinearity
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.run_algorithm import run_algorithm


def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).

    """
    parser = arg_parser(argv, evaluation=evaluation)

    # add custom args here
    parser.add_argument('--my_custom_arg', type=int, default=42, help='Any custom arguments users might define')

    # SampleFactory parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg

class CNNEncoder(EncoderBase):

    def __init__(self, cfg, obs_space, timing):

        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)

        self.conv1 = nn.Conv2d(3, 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 2, stride=1)

        self.nl1 = nonlinearity(cfg)
        self.nl2 = nonlinearity(cfg)

        # Preparing Fully Connected Layers
        conv_layers = [
            self.conv1, self.nl1,
            self.conv2, self.nl2,
        ]

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

        self.encoder_out_size = 512
        self.fc1 = nn.Linear(self.conv_head_out_size,self.encoder_out_size)
        self.nl3 = nonlinearity(cfg)

    def forward(self, x):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        x = self.nl1(self.conv1(x))
        x = self.nl2(self.conv2(x))
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl3(self.fc1(x))
        return x

class RNNEncoder(CNNEncoder):

    def __init__(self, cfg, obs_space,timing):

        super().__init__(cfg,obs_space,timing)
        self.cnn_encoder = CNNEncoder(cfg,obs_space,timing)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict['obs']

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.cnn_encoder(main_obs)
        return x

def register_custom_components():
    register_custom_encoder('retina_encoder', RNNEncoder)
