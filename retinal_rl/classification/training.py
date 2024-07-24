from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from retinal_rl.models.brain import Brain


def run_epoch(
    device: torch.device,
    brain: Brain,
    history: Dict[str, List[float]],
    class_objective: nn.Module,
    recon_objective: nn.Module,
    trainloader: DataLoader[Tuple[Tensor, int]],
    testloader: DataLoader[Tuple[Tensor, int]],
) -> Tuple[Brain, Dict[str, List[float]]]:
    """Perform a single epoch with training and evaluation."""
    brain.train()
    train_losses = process_dataset(
        device, brain, class_objective, recon_objective, trainloader, is_training=True
    )
    brain.eval()
    test_losses = process_dataset(
        device, brain, class_objective, recon_objective, testloader, is_training=False
    )

    # Combine and update history
    for key, value in train_losses.items():
        history[f"train_{key}"].append(value)
    for key, value in test_losses.items():
        history[f"test_{key}"].append(value)

    return brain, history


def process_dataset(
    device: torch.device,
    brain: Brain,
    class_objective: nn.Module,
    recon_objective: nn.Module,
    dataloader: DataLoader[Tuple[Tensor, int]],
    is_training: bool,
) -> Dict[str, float]:
    """Process a dataset (train or test) and return average losses."""
    total_losses: Dict[str, float] = {}
    steps = 0

    for batch in dataloader:
        loss_dict = calculate_loss_dict(
            device, brain, recon_objective, class_objective, batch
        )

        if is_training:
            brain.optimize(loss_dict)

        # Accumulate losses
        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value.item()
        steps += 1

    # Calculate average losses
    return {key: value / steps for key, value in total_losses.items()}


def calculate_loss_dict(
    device: torch.device,
    brain: Brain,
    recon_objective: nn.Module,
    class_objective: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    inputs, classes = batch
    inputs, classes = inputs.to(device), classes.to(device)

    stimuli = {"vision": inputs}

    response = brain(stimuli)
    predicted_classes = response["classifier"]
    reconstructions = response["decoder"]

    class_loss = class_objective(predicted_classes, classes)
    recon_loss = recon_objective(inputs, reconstructions)

    # Calculate the number of correct predictions
    _, predicted_labels = torch.max(predicted_classes, 1)
    fraction_correct = torch.mean((predicted_labels == classes).float())

    return {
        "classification": class_loss,
        "fraction_correct": fraction_correct,
        "reconstruction": recon_loss,
    }
