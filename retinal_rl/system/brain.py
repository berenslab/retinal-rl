"""
retina_rl library

"""

import torch
from torch import nn, Tensor
from collections import OrderedDict
from captum.attr import IntegratedGradients  # ,NoiseTunnel

from sample_factory.model.core import ModelCore
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.typing import Config, ObsSpace, ActionSpace, Dict
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.utils import log
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.model_utils import get_rnn_size
from retinal_rl_models.encoder.vision_models import RetinalModel, RetinalStrideModel, PrototypicalModel

from retinal_rl.util import activation, encoder_out_size


### Registration ###


def register_brain():
    """Registers the retinal model with the global model factory."""
    global_model_factory().register_actor_critic_factory(Brain)


### Model make functions ###


def make_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """defines the encoder constructor."""
    return RetinalEncoder(cfg, obs_space)


def make_core(cfg: Config, core_input_size: int) -> ModelCore:

    if cfg.use_rnn:
        core = LatentRNN(cfg, core_input_size)
    # If using identity activation functions
    elif cfg.activation == "identity":
        core = LatentIdentity(cfg, core_input_size)
    else:
        core = LatentFFN(cfg, core_input_size)

    return core


def make_value_network(cfg: Config, enc, cor, crit):

    if cfg.input_satiety is True:

        if cfg.use_rnn:

            valnet = MeasuredValueRNN(cfg, enc, cor, crit)
            # onnx_inpts = (nobs1,msms1,rnn_states)
        else:
            valnet = MeasuredValueFFN(cfg, enc, cor, crit)
            # onnx_inpts = (nobs1,msms1)

    else:

        if cfg.use_rnn:
            valnet = ValueRNN(cfg, enc, cor, crit)
            # onnx_inpts = (nobs1,rnn_states)

        else:
            valnet = ValueFFN(cfg, enc, cor, crit)
            # onnx_inpts = (nobs1)

    return valnet


### Brain ###


class Brain(ActorCritic):
    def __init__(
        self,
        cfg: Config,
        obs_space: ObsSpace,
        action_space: ActionSpace,
    ):
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = make_encoder(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = make_core(cfg, self.encoder.get_out_size())

        self.decoder = IdentityDecoder(cfg, self.core.get_out_size())

        decoder_out_size: int = self.decoder.get_out_size()

        self.critic_linear = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(
            decoder_out_size
        )

        self.valnet = make_value_network(
            cfg, self.encoder, self.core, self.critic_linear
        )
        self.att_method = IntegratedGradients(self.valnet)

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool
    ) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = (
            self.action_parameterization(decoder_output)
        )

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        result["latent_states"] = x
        return result

    def prune_inputs(self, nobs, msms, rnn_states):

        if self.cfg.use_rnn:
            if self.cfg.input_satiety is True:
                return (nobs, msms, rnn_states)
            else:
                return (nobs, rnn_states)
        else:
            if self.cfg.input_satiety is True:
                return (nobs, msms)
            else:
                return nobs

    def attribute(self, nobs, msms, rnn_states):

        inpts = self.prune_inputs(nobs, msms, rnn_states)

        atts = self.att_method.attribute(inpts, n_steps=200)

        return atts[0]


### Cores ###


class LatentRNN(ModelCore):

    def __init__(self, cfg, input_size):

        super().__init__(cfg)

        self.cfg = cfg

        self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers)

        self.core_output_size = cfg.rnn_size

    def forward(self, head_output, rnn_states):

        is_seq = not torch.is_tensor(head_output)

        if not is_seq:
            head_output = head_output.unsqueeze(0)

        rnn_states = rnn_states.unsqueeze(0)

        x, new_rnn_states = self.core(head_output, rnn_states.contiguous())

        if not is_seq:
            x = x.squeeze(0)

        new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


class LatentFFN(ModelCore):

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states):
        # Apply tanh to head output
        head_output = torch.tanh(head_output)

        return head_output, fake_rnn_states


class LatentIdentity(ModelCore):

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states):

        return head_output, fake_rnn_states


### Decoders ###


class IdentityDecoder(Decoder):

    def __init__(self, cfg, input_size):

        super().__init__(cfg)

        self.cfg = cfg

        self.decoder_out_size = input_size

    def get_out_size(self):
        return self.decoder_out_size

    def forward(self, core_output):
        return core_output


### Encoders ###


class RetinalEncoder(Encoder):

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        inp_shape = obs_space["obs"].shape[1:]
        if cfg.vision_model == "retinal":
            self.vision_model = RetinalModel(
                cfg.base_channels,
                cfg.rnn_size,
                inp_shape,
                cfg.retinal_bottleneck,
                cfg.activation,
            )
        elif cfg.vision_model == "retinal_stride":
            self.vision_model = RetinalStrideModel(
                cfg.base_channels,
                cfg.rnn_size,
                inp_shape,
                cfg.retinal_bottleneck,
                cfg.activation,
            )
        elif cfg.vision_model == "prototypical":
            self.vision_model = PrototypicalModel(
                cfg.rnn_size, inp_shape, cfg.activation
            )
        else:
            raise Exception("Unknown model type: %r", cfg.vision_model)

        self.encoder_out_size = self.vision_model.get_out_size()

        self.measurement_size = 0

        if "measurements" in list(obs_space.keys()):
            self.measurement_size = obs_space["measurements"].shape[0]

        self.nl_fc = activation(cfg.activation)
        self.fc2 = nn.Linear(
            self.encoder_out_size + self.measurement_size, self.encoder_out_size
        )
        self.fc3 = nn.Linear(self.encoder_out_size, self.encoder_out_size)

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        x = self.vision_model(obs_dict["obs"])
        # concatenate x with measurements
        if self.measurement_size > 0:
            x = torch.cat((x, obs_dict["measurements"]), dim=1)

        x = self.nl_fc(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


### Value Networks ###


class MeasuredValueFFN(nn.Module):
    """
    Converts a basic encoder into a feedforward value network that can be easily analyzed by e.g. captum.
    """

    def __init__(self, cfg, enc, cor, crit):

        super().__init__()

        self.cfg = cfg
        self.encoder = enc
        self.core = cor
        self.critic = crit

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        self.fake_rnn_states = torch.zeros(
            [1, get_rnn_size(cfg)], dtype=torch.float32, device=device
        )

    def forward(self, nobs, msms):
        # conv layer 1

        nobs_dict = {"obs": nobs, "measurements": msms}

        x = self.encoder(nobs_dict)
        x, _ = self.core(x, self.fake_rnn_states)
        x = self.critic(x)

        return x


class ValueFFN(nn.Module):
    """
    Converts a basic encoder into a feedforward value network that can be easily analyzed by e.g. captum.
    """

    def __init__(self, cfg, enc, cor, crit):

        super().__init__()

        self.cfg = cfg
        self.encoder = enc
        self.core = cor
        self.critic = crit

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        self.fake_rnn_states = torch.zeros(
            [1, get_rnn_size(cfg)], dtype=torch.float32, device=device
        )

    def forward(self, nobs):
        # conv layer 1

        nobs_dict = {"obs": nobs}

        x = self.encoder(nobs_dict)
        x, _ = self.core(x, self.fake_rnn_states)
        x = self.critic(x)

        return x


class MeasuredValueRNN(nn.Module):
    """
    Converts a basic encoder into a feedforward value network that can be easily analyzed by e.g. captum.
    """

    def __init__(self, cfg, enc, cor, crit):

        super().__init__()

        self.cfg = cfg
        self.encoder = enc
        self.core = cor
        self.critic = crit

    def forward(self, nobs, msms, rnn_states):
        # conv layer 1

        nobs_dict = {"obs": nobs, "measurements": msms}

        x = self.encoder(nobs_dict)
        x, _ = self.core(x, rnn_states)
        x = self.critic(x)

        return x


class ValueRNN(nn.Module):
    """
    Converts a basic encoder into a feedforward value network that can be easily analyzed by e.g. captum.
    """

    def __init__(self, cfg, enc, cor, crit):

        super().__init__()

        self.cfg = cfg
        self.encoder = enc
        self.core = cor
        self.critic = crit

    def forward(self, nobs, rnn_states):
        # conv layer 1

        nobs_dict = {"obs": nobs}

        x = self.encoder(nobs_dict)
        x, _ = self.core(x, rnn_states)
        x = self.critic(x)

        return x
