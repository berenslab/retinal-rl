"""Losses for training models, and the context required to evaluate them."""

from abc import abstractmethod
from typing import Dict, Generic, List, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from retinal_rl.util import camel_to_snake

ContextT = TypeVar("ContextT", bound="BaseContext")


class BaseContext:
    """Base class for all context objects used in the brain model.

    This class provides the common attributes shared across all context types.

    Attributes
    ----------
        responses (Dict[str, Tensor]): The outputs from various parts of the brain model.
        epoch (int): The current training epoch.

    """

    def __init__(
        self,
        sources: Tensor,
        inputs: Tensor,
        responses: Dict[str, Tensor],
        epoch: int,
    ):
        """Initialize the context object with responses, and the current epoch."""
        self.responses = responses
        self.sources = sources
        self.inputs = inputs
        self.epoch = epoch


class Loss(Generic[ContextT]):
    """Base class for losses that can be used to define a multiobjective optimization problem."""

    def __init__(
        self,
        min_epoch: int = 0,
        max_epoch: int = 1,
        target_circuits: List[str] = [],
        weights: List[float] = [],
    ):
        """Initialize the loss with a weight."""
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.target_circuits = target_circuits
        self.weights = weights

    def __call__(self, context: ContextT) -> Tensor:
        """Compute the weighted loss for this loss.

        Args:
        ----
            context (ContextT): Context information for computing losses.

        Returns:
        -------
            Tensor: A tuple containing the weighted loss and the raw loss value.

        """
        return self.compute_value(context)

    def is_training_epoch(self, epoch: int) -> bool:
        """Check if the objective should currently be pursued.

        Args:
        ----
            epoch (int): Current epoch number.

        Returns:
        -------
            bool: True if the objective should continue training, False otherwise.

        """
        if epoch < self.min_epoch:
            return False
        if self.max_epoch < 0:
            return True
        return epoch < self.max_epoch

    @abstractmethod
    def compute_value(self, context: ContextT) -> Tensor:
        """Compute the value for this losses The context dictionary contains the necessary information to compute the loss."""
        pass

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the loss."""
        return camel_to_snake(self.__class__.__name__)


class ReconstructionLoss(Loss[ContextT]):
    """Loss for computing the reconstruction loss between inputs and reconstructions."""

    def __init__(
        self,
        min_epoch: int = 0,
        max_epoch: int = 1,
        target_circuits: List[str] = [],
        weights: List[float] = [],
    ):
        """Initialize the reconstruction loss loss."""
        super().__init__(min_epoch, max_epoch, target_circuits, weights)
        self.loss_fn = nn.MSELoss(reduction="mean")

    def compute_value(self, context: ContextT) -> Tensor:
        """Compute the mean squared error between inputs and reconstructions."""
        sources = context.sources
        reconstructions = context.responses["decoder"]

        if sources.shape != reconstructions.shape:
            raise ValueError(
                f"Shape mismatch: sources {sources.shape}, reconstructions {reconstructions.shape}"
            )

        return self.loss_fn(reconstructions, sources)


class L1Sparsity(Loss[ContextT]):
    """Loss for computing the L1 sparsity of activations."""

    def __init__(
        self,
        target_responses: List[str],
        min_epoch: int = 0,
        max_epoch: int = 1,
        target_circuits: List[str] = [],
        weights: List[float] = [],
    ):
        """Initialize the reconstruction loss loss."""
        super().__init__(min_epoch, max_epoch, target_circuits, weights)

        """Initialize the L1 sparsity loss."""
        self.target_responses = target_responses

    def compute_value(self, context: ContextT) -> Tensor:
        """Compute the L1 sparsity of activations."""
        activations: List[Tensor] = []
        responses = context.responses
        for target in self.target_responses:
            if target not in responses:
                raise ValueError(f"Target {target} not found in responses")
            activations.append(responses[target])
        return torch.mean(torch.stack([act.abs().mean() for act in activations]))


class KLDivergenceSparsity(Loss[ContextT]):
    """Loss for computing the KL divergence sparsity of activations."""

    def __init__(
        self,
        target_responses: List[str],
        target_sparsity: float = 0.05,
        min_epoch: int = 0,
        max_epoch: int = 1,
        target_circuits: List[str] = [],
        weights: List[float] = [],
    ):
        """Initialize the KL divergence sparsity loss."""
        super().__init__(min_epoch, max_epoch, target_circuits, weights)
        self.target_responses = target_responses
        self.target_sparsity = target_sparsity

    def compute_value(self, context: ContextT) -> torch.Tensor:
        """Compute the KL divergence sparsity of activations."""
        responses = context.responses
        activations: List[Tensor] = []
        for target in self.target_responses:
            if target not in responses:
                raise ValueError(f"Target {target} not found in responses")
            activations.append(responses[target])

        kl_divs: List[Tensor] = []
        for act in activations:
            avg_activation = torch.mean(act, dim=0)
            kl_div = self.target_sparsity * torch.log(
                self.target_sparsity / (avg_activation + 1e-8)
            ) + (1 - self.target_sparsity) * torch.log(
                (1 - self.target_sparsity) / (1 - avg_activation + 1e-8)
            )
            kl_divs.append(torch.mean(kl_div))

        return torch.mean(torch.stack(kl_divs))
