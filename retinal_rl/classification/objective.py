"""Objectives for training models."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import BaseContext, Objective


class ClassificationContext(BaseContext):
    """Context class for classification tasks.

    This class extends BaseContext with attributes specific to classification problems.

    Attributes
    ----------
        inputs (Tensor): The input data for the classification task.
        classes (Tensor): The true class labels for the input data.

    """

    def __init__(
        self,
        responses: Dict[str, Tensor],
        epoch: int,
        inputs: Tensor,
        classes: Tensor,
    ):
        """Initialize the classification context object."""
        super().__init__(responses, epoch)
        self.inputs = inputs
        self.classes = classes


class ClassificationObjective(Objective[ClassificationContext]):
    """Objective for computing the cross entropy loss."""

    def __init__(self, weight: float = 1.0):
        """Initialize the classification loss objective."""
        super().__init__(weight)
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_value(self, context: ClassificationContext) -> Tensor:
        """Compute the cross entropy loss between the predictions and the targets."""
        predictions = context.responses["classifier"].detach().requires_grad_(True)
        classes = context.classes

        if predictions.shape[0] != classes.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape}, classes {classes.shape}"
            )

        return self.loss_fn(predictions, classes)


class PercentCorrect(Objective[ClassificationContext]):
    """Objective for computing the percent correct classification."""

    def __init__(self, weight: float = 1.0):
        """Initialize the percent correct classification objective."""
        super().__init__(weight)

    def compute_value(self, context: ClassificationContext) -> Tensor:
        """Compute the percent correct classification."""
        predictions = context.responses["classifier"].detach().requires_grad_(True)
        classes = context.classes
        if predictions.shape[0] != classes.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape}, classes {classes.shape}"
            )
        predicted = torch.argmax(predictions, dim=1)
        correct = (predicted == classes).sum()
        total = torch.tensor(classes.size(0))
        return correct / total


def get_classification_context(
    device: torch.device,
    brain: Brain,
    epoch: int,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> ClassificationContext:
    """Calculate the loss dictionary for a single batch.

    This function processes a single batch of data through the Brain model and prepares
    a context dictionary for the optimizer to calculate losses.

    Args:
    ----
        device (torch.device): The device to run the computations on.
        brain (Brain): The Brain model to process the data.
        epoch (int): The current epoch number.
        batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and labels.

    Returns:
    -------
        Dict[str, torch.Tensor]: A context dictionary containing all necessary information
                                 for loss calculation and optimization.

    """
    inputs, classes = batch
    inputs, classes = inputs.to(device), classes.to(device)

    stimuli = {"vision": inputs}
    responses = brain(stimuli)

    return ClassificationContext(
        responses=responses,
        epoch=epoch,
        inputs=inputs,
        classes=classes,
    )
