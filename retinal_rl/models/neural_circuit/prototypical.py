from collections import OrderedDict

from torch import nn
from torch import Tensor

from retinal_rl.models.neural_circuit import NeuralCircuit


# Prototypical Encoder
class PrototypicalEncoder(NeuralCircuit):
    def __init__(self, latent_size: int, input_shape: list[int], act_name: str):
        super().__init__()

        self.act_name = act_name
        self.latent_size = latent_size
        self.nl_fc = self.str_to_activation(act_name)
        conv_layers = OrderedDict[str, nn.Module](
            [
                ("conv1_filters", nn.Conv2d(3, 16, 3, stride=2, padding=1)),
                ("conv1_output", self.str_to_activation(self.act_name)),
                ("conv2_filters", nn.Conv2d(16, 32, 3, stride=2, padding=1)),
                ("conv2_output", self.str_to_activation(self.act_name)),
                ("conv3_filters", nn.Conv2d(32, 64, 3, stride=2, padding=1)),
                ("conv3_output", self.str_to_activation(self.act_name)),
            ]
        )

        self.conv_head = nn.Sequential(conv_layers)
        self.conv_head_out_size = self.calc_num_elements(self.conv_head, input_shape)
        self.fc1 = nn.Linear(self.conv_head_out_size, self.latent_size)

    def forward(self, x: Tensor):
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

    def get_out_size(self) -> int:
        return self.latent_size


# Prototypical Decoder
class PrototypicalDecoder(NeuralCircuit):
    def __init__(self, latent_size: int, input_shape: list[int], act_name: str):
        super().__init__()

        self.act_name = act_name
        self.input_sahpe = input_shape
        self.latent_size = latent_size

        # Define the fully connected layer that will reshape the latent vector
        # into a feature map of appropriate size for deconvolution
        self.fc1 = nn.Linear(self.latent_size, 64 * 8 * 8)

        # Activation function
        self.nl_fc = self.str_to_activation(self.act_name)

        # Define the deconvolutional layers to upsample the feature map to the original image size
        deconv_layers = OrderedDict[str, nn.Module](
            [
                (
                    "deconv1_filters",
                    nn.ConvTranspose2d(
                        64, 32, 3, stride=2, padding=1, output_padding=1
                    ),
                ),
                ("deconv1_output", self.str_to_activation(self.act_name)),
                (
                    "deconv2_filters",
                    nn.ConvTranspose2d(
                        32, 16, 3, stride=2, padding=1, output_padding=1
                    ),
                ),
                ("deconv2_output", self.str_to_activation(self.act_name)),
                ("deconv3_filters", nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)),
                (
                    "deconv3_output",
                    nn.Sigmoid(),
                ),  # Assuming output is an image scaled between 0 and 1
            ]
        )

        self.deconv_head = nn.Sequential(deconv_layers)

    def forward(self, x: Tensor):
        x = self.nl_fc(self.fc1(x))
        x = x.view(-1, 64, 8, 8)  # Reshape to (batch_size, 64, 8, 8)
        x = self.deconv_head(x)
        return x
