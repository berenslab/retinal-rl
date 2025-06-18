"""Minimal variational neural circuits for VAE implementation."""

import logging

import torch
from torch import Tensor, nn

# Assuming this import from your framework
from retinal_rl.models.neural_circuit import NeuralCircuit

logger = logging.getLogger(__name__)


class VariationalBottleneck(NeuralCircuit):
    """
    Variational bottleneck that produces distribution parameters for VAE.

    Takes encoder output and produces mean (mu) and log-variance (log_var)
    parameters for the latent distribution, concatenated for compatibility
    with single-output circuit design.

    Args:
        input_shape: Shape of input tensor from encoder
        latent_dim: Dimensionality of the latent space
    """

    def __init__(
        self,
        input_shape: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__(input_shape)

        self.latent_dim = latent_dim

        # Calculate flattened input size
        self.input_size = int(torch.prod(torch.tensor(input_shape)))

        # Separate heads for mean and log-variance
        self.fc_mu = nn.Linear(self.input_size, latent_dim)
        self.fc_log_var = nn.Linear(self.input_size, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass producing concatenated [mu, log_var].

        Args:
            x: Input tensor [batch_size, *input_shape]

        Returns:
            Concatenated [batch_size, 2 * latent_dim] containing [mu, log_var]
        """
        # Flatten input
        x = x.view(x.size(0), -1)

        # Compute distribution parameters
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        # Concatenate for single output
        return torch.cat([mu, log_var], dim=1)

    @property
    def output_shape(self) -> list[int]:
        """Output shape is [2 * latent_dim]."""
        return [2 * self.latent_dim]


class ReparameterizationSampler(NeuralCircuit):
    """
    Samples from latent distribution using reparameterization trick.

    Takes concatenated distribution parameters and samples latent codes.

    Args:
        input_shape: Should be [2 * latent_dim] from VariationalBottleneck
    """

    def __init__(
        self,
        input_shape: list[int],
    ) -> None:
        super().__init__(input_shape)

        if len(input_shape) != 1 or input_shape[0] % 2 != 0:
            raise ValueError(
                f"ReparameterizationSampler expects input_shape [2*latent_dim], "
                f"got {input_shape}"
            )

        self.latent_dim = input_shape[0] // 2

    def forward(self, x: Tensor) -> Tensor:
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            x: Concatenated parameters [batch_size, 2 * latent_dim]

        Returns:
            Sampled latent codes [batch_size, latent_dim]
        """
        # Split concatenated parameters
        mu, log_var = torch.split(x, self.latent_dim, dim=1)

        if self.training:
            # Reparameterization trick: z = mu + sigma * epsilon
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # During evaluation, use mean
            z = mu

        return z

    @property
    def output_shape(self) -> list[int]:
        """Output shape is [latent_dim]."""
        return [self.latent_dim]
