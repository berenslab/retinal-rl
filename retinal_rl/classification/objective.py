"""Objectives for training models."""

from typing import Any, Dict

import torch.nn as nn
from torch import Tensor

from retinal_rl.models.objective import Objective


class ClassificationObjective(Objective):
    """Objective for computing the cross entropy loss."""

    def __init__(self, weight: float = 1.0):
        """Initialize the classification loss objective."""
        super().__init__(weight)
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_value(self, context: Dict[str, Any]) -> Tensor:
        """Compute the cross entropy loss between the predictions and the targets."""
        predictions = context["predictions"]
        classes = context["classes"]

        if predictions.shape[0] != classes.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape}, classes {classes.shape}"
            )

        return self.loss_fn(predictions, classes)


class PercentCorrect(Objective):
    """Objective for computing the percent correct classification."""

    def __init__(self, weight: float = 1.0):
        """Initialize the percent correct classification objective."""
        super().__init__(weight)

    def compute_value(self, context: Dict[str, Any]) -> Tensor:
        """Compute the percent correct classification."""
        predictions = context["predictions"]
        classes = context["classes"]
        if predictions.shape[0] != classes.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape}, classes {classes.shape}"
            )
        _, predicted = predictions.max(1)
        correct = (predicted == classes).sum()
        total = classes.size(0)
        return correct / total
