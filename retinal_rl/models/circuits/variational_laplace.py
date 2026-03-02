"""Variational neural circuits for VAE implementation."""

import logging

import torch
from beartype import beartype
from torch import Tensor
from torch.distributions import Laplace

from retinal_rl.models.neural_circuit import NeuralCircuit

logger = logging.getLogger(__name__)


class VariationalBottleneckLaplace(NeuralCircuit):
    """
    Variational bottleneck that samples from latent distribution using reparameterization trick.

    Takes two inputs (mu and log_b from separate circuits) and produces sampled latent codes.
    Works with both fully connected (1D) and convolutional (multi-dimensional) latent spaces.

    Args:
        input_shapes: List containing [mu_shape, log_var_shape] - should be identical
    """

    def __init__(
        self,
        input_shapes: list[list[int]],
    ) -> None:
        super().__init__(input_shapes)

        if len(input_shapes) != 2:
            raise ValueError(
                f"VariationalBottleneck expects 2 inputs (mu, log_var), "
                f"got {len(input_shapes)}"
            )

        if input_shapes[0] != input_shapes[1]:
            raise ValueError(
                f"mu and log_var must have same shape, got {input_shapes[0]} and {input_shapes[1]}"
            )

        self.latent_shape = input_shapes[0]

    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            inputs: Tuple containing (mu, log_var) tensors with identical shapes

        Returns:
            Tuple containing sampled latent codes with same shape as inputs
        """
        if len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs (mu, log_var), got {len(inputs)}")

        mu, log_b = inputs # log_b is the log of the scale parameter for the Laplace distribution
        
        #here we do laplace reparameterization using the inverse CDF (quantile function) of the Laplace distribution
        if self.training:
            b = torch.exp(log_b)
            dist = Laplace(mu, b)
            z = dist.rsample()  # rsample() allows for backpropagation through the sampling process
        else:
            # During evaluation, use mean
            z = mu

        return (z,)

    @property
    def output_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Output shape matches the input latent shape."""
        return (tuple(self.latent_shape),)

