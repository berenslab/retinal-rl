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
        responses (dict[str, tuple[Tensor, ...]]): The outputs from various parts of the brain model.
        sources (Tensor): The original source data before any transformations.
        inputs (Tensor): The processed input data fed to the model.
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
        """Compute the value for this statistic. The context contains the necessary information to compute the value."""
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
        """Initialize the reconstruction loss.

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
        reconstructions = context.responses[self.target_decoder][
            self.decoder_output_index
        ]

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

class LaplaceReconstructionLoss(ReconstructionLoss[ContextT]):
    """L1 reconstruction loss for Laplace likelihood p(x|z).

    Uses L1 (MAE) loss which is the negative log-likelihood of a Laplace distribution.
    This is more robust to outliers and preserves sharp edges compared to MSE.

    Args:
        target_decoder: Name of the decoder circuit producing reconstructions
        target_circuits: Circuits to optimize with this loss
        weights: Weights for each circuit
        min_epoch: Minimum epoch to start applying this loss
        max_epoch: Maximum epoch to stop applying this loss
        normalize: Whether to normalize inputs and reconstructions
        decoder_output_index: Which output index to use from decoder circuit tuple (default: 0)
        learn_scale: If True, expects decoder to output both mu and log_b (full Laplace likelihood)
        scale_output_index: Which output index contains log_b values (if learn_scale=True)
    """

    def __init__(
        self,
        target_decoder: str,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        normalize: bool = False,
        decoder_output_index: int = 0,
        learn_scale: bool = False,
        scale_output_index: int = 1,
    ):
        """Initialize the Laplace reconstruction loss."""
        super().__init__(
            target_decoder, target_circuits, weights, min_epoch, max_epoch,
            normalize, decoder_output_index,
        )
        self.learn_scale = learn_scale
        self.scale_output_index = scale_output_index

    def compute_value(self, context: ContextT) -> Tensor:
        """Compute the L1 loss (Laplace negative log-likelihood).

        If learn_scale=False:
            Loss = ||x - x_recon||_1 (simple L1 loss)

        If learn_scale=True:
            Loss = log(2b) + |x - mu| / b (full Laplace NLL)
        """
        sources = context.sources
        decoder_outputs = context.responses[self.target_decoder]

        # Get reconstructions (mu in Laplace terms)
        reconstructions = decoder_outputs[self.decoder_output_index]

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

        if not self.learn_scale:
            # Simple L1 loss (assumes constant scale absorbed into weight)
            return torch.nn.functional.l1_loss(reconstructions, sources, reduction="mean")
        # Full Laplace negative log-likelihood with learnable scale
        if len(decoder_outputs) <= self.scale_output_index:
            raise ValueError(
                f"learn_scale=True but decoder only has {len(decoder_outputs)} outputs. "
                f"Expected log_b at index {self.scale_output_index}"
            )

        log_b = decoder_outputs[self.scale_output_index]
        b = torch.exp(log_b)  # Ensure positive scale

        # Negative log-likelihood: log(2b) + |x - mu| / b
        nll = torch.log(2 * b) + torch.abs(sources - reconstructions) / b
        return nll.mean()

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss."""
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

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss, including the target response."""
        return f"kl_divergence_sparsity_{self.target_response.lower()}"


class KLDivergenceLoss(Loss[ContextT]):
    """
    KL divergence loss between learned posterior q(z|x) and prior p(z).

    Assumes standard normal prior N(0, I). Gets distribution parameters
    from separate mu and log_var circuit outputs.

    Uses the unified weighting system - set weights parameter for KL term scaling
    (equivalent to beta-VAE parameter in literature).

    Args:
        target_mu: Name of the circuit producing mu values
        target_logvar: Name of the circuit producing log_var values
        target_circuits: Circuits to optimize with this loss
        weights: Weights for each circuit (controls KL term strength)
        min_epoch: Minimum epoch to start applying this loss
        max_epoch: Maximum epoch to stop applying this loss
        mu_output_index: Which output index to use from mu circuit tuple (default: 0)
        logvar_output_index: Which output index to use from log_var circuit tuple (default: 0)
    """

    def __init__(
        self,
        target_mu: str,
        target_logvar: str,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        mu_output_index: int = 0,
        logvar_output_index: int = 0,
    ) -> None:
        """
        Initialize KL divergence loss with unified weighting system.

        Args:
            target_mu: Name of circuit producing mu values
            target_logvar: Name of circuit producing log_var values
            target_circuits: Circuits to optimize with this loss
            weights: Loss weights for each circuit (controls KL strength, replaces beta)
            min_epoch: Minimum epoch to start applying this loss
            max_epoch: Maximum epoch to stop applying this loss
            mu_output_index: Which output index to use from mu circuit tuple (default: 0)
            logvar_output_index: Which output index to use from logvar circuit tuple (default: 0)
        """
        super().__init__(target_circuits, weights, min_epoch, max_epoch)

        self.target_mu = target_mu
        self.target_logvar = target_logvar
        self.mu_output_index = mu_output_index
        self.logvar_output_index = logvar_output_index

    def compute_value(self, context: ContextT) -> Tensor:
        """
        Compute KL divergence loss.

        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

        Args:
            context: Training context containing brain and responses

        Returns:
            KL divergence loss value
        """
        # Get mu and log_var outputs from separate circuits
        if self.target_mu not in context.responses:
            raise ValueError(
                f"Target mu circuit '{self.target_mu}' not found in responses. "
                "Make sure it's connected in the brain's connectome."
            )
        if self.target_logvar not in context.responses:
            raise ValueError(
                f"Target logvar circuit '{self.target_logvar}' not found in responses. "
                "Make sure it's connected in the brain's connectome."
            )

        mu = context.responses[self.target_mu][self.mu_output_index]
        log_var = context.responses[self.target_logvar][self.logvar_output_index]

        # Compute KL divergence
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        # Sum over latent dimensions, mean over batch
        return kl_elementwise.sum(dim=1).mean()

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss."""
        return f"kl_divergence_{self.target_mu}_{self.target_logvar}"

class LaplaceKLDivergenceLoss(Loss[ContextT]):
    """
    KL divergence loss between Laplace posterior q(z|x) and Laplace prior p(z).

    Assumes standard Laplace prior Laplace(0, prior_b). Gets distribution parameters
    from separate mu and log_b circuit outputs.

    The closed-form KL divergence between two Laplace distributions is:
    KL(q||p) = (b_q/b_p) * exp(-|mu_q - mu_p|/b_q) + |mu_q - mu_p|/b_p + log(b_p/b_q) - 1

    For standard prior p(z) = Laplace(0, prior_b):
    KL(q||p) = (b/prior_b) * exp(-|mu|/b) + |mu|/prior_b + log(prior_b/b) - 1

    Args:
        target_mu: Name of the circuit producing mu (location) values
        target_log_b: Name of the circuit producing log_var as is the terminology in rest of the doc (log scale) values
        prior_b: Scale parameter for prior Laplace(0, prior_b) distribution (default: 1.0)
        target_circuits: Circuits to optimize with this loss
        weights: Weights for each circuit (controls KL term strength, like beta in beta-VAE)
        min_epoch: Minimum epoch to start applying this loss
        max_epoch: Maximum epoch to stop applying this loss
        mu_output_index: Which output index to use from mu circuit tuple (default: 0)
        log_b_output_index: Which output index to use from log_b circuit tuple (default: 0)
    """

    def __init__(
        self,
        target_mu: str,
        target_log_b: str,
        prior_b: float = 1.0,
        target_circuits: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        min_epoch: Optional[int] = None,
        max_epoch: Optional[int] = None,
        mu_output_index: int = 0,
        log_b_output_index: int = 0,
    ) -> None:
        """
        Initialize Laplace KL divergence loss.

        Args:
            target_mu: Name of circuit producing mu (location) values
            target_log_b: Name of circuit producing log_b (log scale) values
            prior_b: Scale parameter of prior Laplace(0, prior_b) distribution
            target_circuits: Circuits to optimize with this loss
            weights: Loss weights for each circuit (controls KL strength)
            min_epoch: Minimum epoch to start applying this loss
            max_epoch: Maximum epoch to stop applying this loss
            mu_output_index: Which output index to use from mu circuit tuple
            log_b_output_index: Which output index to use from log_b circuit tuple
        """
        super().__init__(target_circuits, weights, min_epoch, max_epoch)

        self.target_mu = target_mu
        self.target_log_b = target_log_b
        self.prior_b = prior_b
        self.mu_output_index = mu_output_index
        self.log_b_output_index = log_b_output_index

    def compute_value(self, context: ContextT) -> Tensor:
        """
        Compute KL divergence between Laplace posterior and prior.

        For q(z|x) = Laplace(mu, b) and p(z) = Laplace(0, prior_b):
        KL(q||p) = (b/prior_b) * exp(-|mu|/b) + |mu|/prior_b + log(prior_b/b) - 1

        Args:
            context: Training context containing brain and responses

        Returns:
            KL divergence loss value
        """
        # Get mu and log_b outputs from separate circuits
        if self.target_mu not in context.responses:
            raise ValueError(
                f"Target mu circuit '{self.target_mu}' not found in responses. "
                "Make sure it's connected in the brain's connectome."
            )
        if self.target_log_b not in context.responses:
            raise ValueError(
                f"Target log_b circuit '{self.target_log_b}' not found in responses. "
                "Make sure it's connected in the brain's connectome."
            )

        mu = context.responses[self.target_mu][self.mu_output_index]
        log_b = context.responses[self.target_log_b][self.log_b_output_index]

        # Ensure b is positive by exponentiating log_b
        b = torch.exp(log_b)

        # Compute KL divergence: KL(Laplace(mu, b) || Laplace(0, prior_b))
        # KL = (b/prior_b) * exp(-|mu|/b) + |mu|/prior_b + log(prior_b/b) - 1
        kl_elementwise = (
            (b / self.prior_b) * torch.exp(-torch.abs(mu) / b) +  # Exponential term
            torch.abs(mu) / self.prior_b +                         # L1 penalty on mu (sparsity!)
            torch.log(torch.tensor(self.prior_b, device=b.device)) - log_b - 1  # Log ratio term
        )

        # Sum over latent dimensions, mean over batch
        return kl_elementwise.sum(dim=1).mean()

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss."""
        return f"kl_divergence_{self.target_mu}_{self.target_log_b}"
