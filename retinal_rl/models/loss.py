"""Losses for training models, and the context required to evaluate them."""

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import torch
from torch import Tensor, nn

from retinal_rl.util import camel_to_snake

ContextT = TypeVar("ContextT", bound="BaseContext")


class BaseContext:
    """Base class for all context objects used in the brain model.

    This class provides the common attributes shared across all context types.

    Attributes
    ----------
        responses (dict[str, Tensor]): The outputs from various parts of the brain model.
        epoch (int): The current training epoch.

    """

    def __init__(
        self,
        sources: Tensor,
        inputs: Tensor,
        responses: dict[str, tuple[Tensor, ...]],
        epoch: int,
    ):
        """Initialize the context object with responses, and the current epoch."""
        self.responses = responses
        self.sources = sources
        self.inputs = inputs
        self.epoch = epoch


class LoggingStatistic(Generic[ContextT]):
    """Base class for statistics that should be logged."""

    def __call__(self, context: ContextT) -> Tensor:
        return self.compute_value(context)

    @abstractmethod
    def compute_value(self, context: ContextT) -> Tensor:
        """Compute the value for this losses The context dictionary contains the necessary information to compute the loss."""
        pass

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss."""
        return camel_to_snake(self.__class__.__name__)


class Loss(LoggingStatistic[ContextT]):
    """Base class for losses that can be used to define a multiobjective optimization problem.

    Attributes
    ----------
        target_circuits (list[str]): The target circuits for the loss. If '__all__', will target all circuits
        weights (list[float]): The weights for the loss.
        min_epoch (int): The minimum epoch to start training the loss.
        max_epoch (int): The maximum epoch to train the loss. Unbounded if < 0.

    """

    def __init__(
        self,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float] | float | int] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
    ):
        """Initialize the loss with a weight."""
        if target_circuits is None:
            target_circuits = []
        if weights is None:
            weights = [1.0] * len(target_circuits)
        elif isinstance(weights, float):
            weights = [weights]
        elif isinstance(weights, int):
            weights = [float(weights)] * len(target_circuits)

        self.target_circuits = target_circuits
        self.weights = weights
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch

    def is_training_epoch(self, epoch: int) -> bool:
        """Check if the objective should currently be pursued.

        Args:
        ----
            epoch (int): Current epoch number.

        Returns:
        -------
            bool: True if the objective should continue training, False otherwise.

        """
        if self.min_epoch is not None and epoch < self.min_epoch:
            return False
        return self.max_epoch is None or epoch <= self.max_epoch


class ReconstructionLoss(Loss[ContextT]):
    """Loss for computing the reconstruction loss between inputs and reconstructions."""

    def __init__(
        self,
        target_decoder: str,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        normalize: bool = False,
        decoder_output_index: int = 0,
    ):
        """Initialize the reconstruction loss loss.
        
        Args:
            decoder_output_index: Which output index to use from decoder circuit tuple (default: 0)
        """
        super().__init__(target_circuits, weights, min_epoch, max_epoch)
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.target_decoder = target_decoder
        self.normalize = normalize
        self.decoder_output_index = decoder_output_index

    def compute_value(self, context: ContextT) -> Tensor:
        """Compute the mean squared error between inputs and reconstructions."""
        sources = context.sources
        reconstructions = context.responses[self.target_decoder][self.decoder_output_index]

        if sources.shape != reconstructions.shape:
            raise ValueError(
                f"Shape mismatch: sources {sources.shape}, reconstructions {reconstructions.shape}"
            )

        if self.normalize:
            sources = (
                sources - sources.mean(dim=[1, 2, 3])[:, None, None, None]
            ) / sources.std(dim=[1, 2, 3])[:, None, None, None]
            reconstructions = (
                reconstructions
                - reconstructions.mean(dim=[1, 2, 3])[:, None, None, None]
            ) / reconstructions.std(dim=[1, 2, 3])[:, None, None, None]

        return self.loss_fn(reconstructions, sources)

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss, including the target decoder."""
        return f"reconstruction_loss_{self.target_decoder.lower()}"


class L1Sparsity(Loss[ContextT]):
    """Loss for computing the L1 sparsity of activations."""

    def __init__(
        self,
        target_response: str,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        response_output_index: int = 0,
    ):
        """Initialize the L1 sparsity loss.
        
        Args:
            response_output_index: Which output index to use from target response circuit tuple (default: 0)
        """
        super().__init__(target_circuits, weights, min_epoch, max_epoch)

        self.target_response = target_response
        self.response_output_index = response_output_index

    def compute_value(self, context: ContextT) -> Tensor:
        """Compute the L1 sparsity of activations."""
        responses = context.responses
        if self.target_response not in responses:
            raise ValueError(f"Target {self.target_response} not found in responses")
        activation = responses[self.target_response][self.response_output_index]
        return torch.mean(activation.abs().mean())

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss, including the target response."""
        return f"l1_sparsity_{self.target_response.lower()}"


class KLDivergenceSparsity(Loss[ContextT]):
    """Loss for computing the KL divergence sparsity of activations."""

    def __init__(
        self,
        target_response: str,
        target_sparsity: float = 0.05,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        response_output_index: int = 0,
    ):
        """Initialize the KL divergence sparsity loss.
        
        Args:
            response_output_index: Which output index to use from target response circuit tuple (default: 0)
        """
        super().__init__(target_circuits, weights, min_epoch, max_epoch)
        self.target_response = target_response
        self.target_sparsity = target_sparsity
        self.response_output_index = response_output_index

    def compute_value(self, context: ContextT) -> torch.Tensor:
        """Compute the KL divergence sparsity of activations."""
        responses = context.responses
        if self.target_response not in responses:
            raise ValueError(f"Target {self.target_response} not found in responses")
        activation = responses[self.target_response][self.response_output_index]

        avg_activation = torch.mean(activation, dim=0)
        kl_div = self.target_sparsity * torch.log(
            self.target_sparsity / (avg_activation + 1e-8)
        ) + (1 - self.target_sparsity) * torch.log(
            (1 - self.target_sparsity) / (1 - avg_activation + 1e-8)
        )
        return torch.mean(kl_div)


class KLDivergenceLoss(Loss[ContextT]):
    """
    KL divergence loss between learned posterior q(z|x) and prior p(z).

    Assumes standard normal prior N(0, I). Gets distribution parameters
    from the VariationalBottleneck circuit output.

    Args:
        target_bottleneck: Name of the VariationalBottleneck circuit
        beta: Weight for KL term (beta-VAE parameter, default=1.0)
        target_circuits: Circuits to optimize with this loss
        weights: Weights for each circuit
        min_epoch: Minimum epoch to start applying this loss
        max_epoch: Maximum epoch to stop applying this loss
    """

    def __init__(
        self,
        target_bottleneck: str,
        beta: float = 1.0,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        bottleneck_output_index: int = 0,
    ) -> None:
        """
        Args:
            bottleneck_output_index: Which output index to use from bottleneck circuit tuple (default: 0)
        """
        super().__init__(target_circuits, weights, min_epoch, max_epoch)

        self.target_bottleneck = target_bottleneck
        self.beta = beta
        self.bottleneck_output_index = bottleneck_output_index

    def compute_value(self, context: ContextT) -> Tensor:
        """
        Compute KL divergence loss.

        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

        Args:
            context: Training context containing brain and responses

        Returns:
            KL divergence loss value
        """
        # Get the bottleneck output
        if self.target_bottleneck not in context.responses:
            raise ValueError(
                f"Target bottleneck '{self.target_bottleneck}' not found in responses. "
                "Make sure it's connected in the brain's connectome."
            )

        bottleneck_output = context.responses[self.target_bottleneck][self.bottleneck_output_index]

        # Split concatenated mu and log_var
        latent_dim = bottleneck_output.size(1) // 2
        mu, log_var = torch.split(bottleneck_output, latent_dim, dim=1)

        # Compute KL divergence
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        # Sum over latent dimensions, mean over batch
        kl = kl_elementwise.sum(dim=1).mean()

        # Apply beta weighting
        return self.beta * kl

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss."""
        return f"kl_divergence_{self.target_bottleneck}"
