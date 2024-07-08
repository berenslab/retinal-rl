### Imports ###

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from retinal_rl.models.brain import Brain


def run_epoch(
    device: torch.device,
    brain: Brain,
    history: dict[str, List[float]],
    recon_weight: float,
    optimizer: Optimizer,
    class_objective: nn.Module,
    recon_objective: nn.Module,
    trainloader: DataLoader[Tuple[Tensor, int]],
    testloader: DataLoader[Tuple[Tensor, int]],
) -> Tuple[Brain, dict[str, List[float]]]:
    """Perform a single run with a train/test split."""
    brain.train()  # Ensure the model is in training mode

    train_loss, train_class_loss, train_frac_correct, train_recon_loss = train_epoch(
        device,
        brain,
        optimizer,
        recon_weight,
        recon_objective,
        class_objective,
        trainloader,
    )
    test_loss, test_class_loss, test_frac_correct, test_recon_loss = evaluate_model(
        device,
        brain,
        recon_weight,
        recon_objective,
        class_objective,
        testloader,
    )

    history["train_total"].append(train_loss)
    history["train_classification"].append(train_class_loss)
    history["train_fraction_correct"].append(train_frac_correct)
    history["train_reconstruction"].append(train_recon_loss)
    history["test_total"].append(test_loss)
    history["test_classification"].append(test_class_loss)
    history["test_fraction_correct"].append(test_frac_correct)
    history["test_reconstruction"].append(test_recon_loss)

    return brain, history


def calculate_loss(
    device: torch.device,
    brain: Brain,
    recon_weight: float,
    recon_objective: nn.Module,
    class_objective: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, classes = batch
    inputs, classes = inputs.to(device), classes.to(device)

    stimuli = {"vision": inputs}

    response = brain(stimuli)
    predicted_classes = response["classifier"]
    reconstructions = response["decoder"]

    class_loss = class_objective(predicted_classes, classes)
    recon_loss = recon_objective(inputs, reconstructions)

    class_weight = 1 - recon_weight
    loss = class_weight * class_loss + recon_weight * recon_loss

    # Calculate the number of correct predictions
    _, predicted_labels = torch.max(predicted_classes, 1)
    fraction_correct = torch.mean((predicted_labels == classes).float())

    return loss, class_loss, fraction_correct, recon_loss


def train_epoch(
    device: torch.device,
    brain: Brain,
    optimizer: Optimizer,
    recon_weight: float,
    recon_objective: nn.Module,
    class_objective: nn.Module,
    trainloader: DataLoader[Tuple[Tensor, int]],
) -> Tuple[float, float, float, float]:
    """Trains the model for one epoch.

    Returns
    -------
        Tuple[float, float,float]: A tuple containing the average loss,
        reconstruction loss, and classification loss for the epoch.

    """
    losses: dict[str, float]
    losses = {
        "total": 0,
        "classification": 0,
        "fraction_correct": 0,
        "reconstruction": 0,
    }
    steps = 0

    for batch in trainloader:
        optimizer.zero_grad()

        loss, class_loss, frac_correct, recon_loss = calculate_loss(
            device, brain, recon_weight, recon_objective, class_objective, batch
        )

        losses["total"] += loss.item()
        losses["classification"] += class_loss.item()
        losses["fraction_correct"] += frac_correct.item()
        losses["reconstruction"] += recon_loss.item()
        steps += 1

        loss.backward()
        optimizer.step()

    avg_loss = losses["total"] / steps
    avg_class_loss = losses["classification"] / steps
    avg_frac_correct = losses["fraction_correct"] / steps
    avg_recon_loss = losses["reconstruction"] / steps
    return avg_loss, avg_class_loss, avg_frac_correct, avg_recon_loss


def evaluate_model(
    device: torch.device,
    brain: Brain,
    recon_weight: float,
    recon_objective: torch.nn.Module,
    class_objective: torch.nn.Module,
    testloader: DataLoader[Tuple[Tensor, int]],
) -> Tuple[float, float, float, float]:
    brain.eval()  # Ensure the model is in evaluation mode

    losses: dict[str, float]
    losses = {
        "total": 0,
        "classification": 0,
        "fraction_correct": 0,
        "reconstruction": 0,
    }
    steps = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch in testloader:
            loss, class_loss, frac_correct, recon_loss = calculate_loss(
                device, brain, recon_weight, recon_objective, class_objective, batch
            )

            losses["total"] += loss.item()
            losses["classification"] += class_loss.item()
            losses["fraction_correct"] += frac_correct.item()
            losses["reconstruction"] += recon_loss.item()
            steps += 1

        avg_loss = losses["total"] / steps
        avg_class_loss = losses["classification"] / steps
        avg_frac_correct = losses["fraction_correct"] / steps
        avg_recon_loss = losses["reconstruction"] / steps
        return avg_loss, avg_class_loss, avg_frac_correct, avg_recon_loss
