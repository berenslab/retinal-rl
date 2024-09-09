"""Objectives for training models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from retinal_rl.util import camel_to_snake


class Objective(ABC):
    """Base class for objectives that can be used to train a model."""

    def __init__(self, weight: float = 1.0):
        """Initialize the objective with a weight."""
        self.weight = weight

    def __call__(self, context: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """Compute the weighted loss for this objective.

        Args:
        ----
            context (Dict[str, Any]): Context information for computing objectives.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the weighted loss and the raw loss value.

        """
        value = self.compute_value(context)
        return (self.weight * value, value)

    @abstractmethod
    def compute_value(self, context: Dict[str, Any]) -> Tensor:
        """Compute the value for this objective. The context dictionary contains the necessary information to compute the objective."""
        pass

    @property
    def key_name(self) -> str:
        """Return a user-friendly name for the objective."""
        return camel_to_snake(self.__class__.__name__)


class ReconstructionObjective(Objective):
    """Objective for computing the reconstruction loss between inputs and reconstructions."""

    def __init__(self, weight: float = 1.0):
        """Initialize the reconstruction loss objective."""
        super().__init__(weight)
        self.loss_fn = nn.MSELoss(reduction="mean")

    def compute_value(self, context: Dict[str, Any]) -> Tensor:
        """Compute the mean squared error between inputs and reconstructions."""
        inputs = context["responses"]["vision"]
        reconstructions = context["responses"]["decoder"]

        if inputs.shape != reconstructions.shape:
            raise ValueError(
                f"Shape mismatch: inputs {inputs.shape}, reconstructions {reconstructions.shape}"
            )

        return self.loss_fn(reconstructions, inputs)


class L2WeightRegularizer(Objective):
    """Objective for computing the L2 norm of model weights."""

    def __init__(self, weight: float = 1.0):
        """Initialize the L2 weight regularizer."""
        super().__init__(weight)

    def compute_value(self, context: Dict[str, Any]) -> torch.Tensor:
        """Compute the L2 norm of model weights."""
        parameters = context["parameters"]
        if not parameters:
            raise ValueError("No parameters found in context")

        l2_norms = [p.pow(2).mean() for p in parameters]
        return torch.mean(torch.stack(l2_norms))


class L1Sparsity(Objective):
    """Objective for computing the L1 sparsity of activations."""

    def __init__(self, weight: float, targets: List[str]):
        """Initialize the L1 sparsity objective."""
        self.targets = targets
        super().__init__(weight)

    def compute_value(self, context: Dict[str, Any]) -> Tensor:
        """Compute the L1 sparsity of activations."""
        activations: List[Tensor] = []
        responses = context["responses"]
        for target in self.targets:
            if target not in responses:
                raise ValueError(f"Target {target} not found in responses")
            activations.append(responses[target])
        return torch.mean(torch.stack([act.abs().mean() for act in activations]))


class KLDivergenceSparsity(Objective):
    """Objective for computing the KL divergence sparsity of activations."""

    def __init__(self, weight: float, targets: List[str], target_sparsity: float = 0.05):
        """Initialize the KL divergence sparsity objective."""
        self.targets = targets
        self.target_sparsity = target_sparsity
        super().__init__(weight)

    def compute_value(self, context: Dict[str, Any]) -> torch.Tensor:
        """Compute the KL divergence sparsity of activations."""
        responses = context["responses"]
        activations: List[Tensor] = []
        for target in self.targets:
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
