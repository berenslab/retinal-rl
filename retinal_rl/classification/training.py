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

    train_loss, train_recon_loss, train_class_loss = train_epoch(
        device,
        brain,
        optimizer,
        recon_weight,
        recon_objective,
        class_objective,
        trainloader,
    )
    test_loss, test_recon_loss, test_class_loss = evaluate_model(
        device,
        brain,
        recon_weight,
        recon_objective,
        class_objective,
        testloader,
    )

    history["train_total"].append(train_loss)
    history["train_classification"].append(train_class_loss)
    history["train_reconstruction"].append(train_recon_loss)
    history["test_total"].append(test_loss)
    history["test_classification"].append(test_class_loss)
    history["test_reconstruction"].append(test_recon_loss)

    return brain, history


def calculate_loss(
    device: torch.device,
    brain: Brain,
    recon_weight: float,
    recon_objective: nn.Module,
    class_objective: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return loss, class_loss, recon_loss


def train_epoch(
    device: torch.device,
    brain: Brain,
    optimizer: Optimizer,
    recon_weight: float,
    recon_objective: nn.Module,
    class_objective: nn.Module,
    trainloader: DataLoader[Tuple[Tensor, int]],
) -> Tuple[float, float, float]:
    """Trains the model for one epoch.

    Returns
    -------
        Tuple[float, float,float]: A tuple containing the average loss,
        reconstruction loss, and classification loss for the epoch.

    """
    losses: dict[str, List[torch.Tensor]]
    losses = {"total": [], "classification": [], "reconstruction": []}

    for batch in trainloader:
        optimizer.zero_grad()

        loss, class_loss, recon_loss = calculate_loss(
            device, brain, recon_weight, recon_objective, class_objective, batch
        )

        losses["total"].append(loss)
        losses["classification"].append(class_loss)
        losses["reconstruction"].append(recon_loss)

        loss.backward()
        optimizer.step()

    avg_loss = torch.mean(torch.stack(losses["total"])).item()
    avg_class_loss = torch.mean(torch.stack(losses["classification"])).item()
    avg_recon_loss = torch.mean(torch.stack(losses["reconstruction"])).item()
    return avg_loss, avg_class_loss, avg_recon_loss


def evaluate_model(
    device: torch.device,
    brain: Brain,
    recon_weight: float,
    recon_objective: torch.nn.Module,
    class_objective: torch.nn.Module,
    testloader: DataLoader[Tuple[Tensor, int]],
) -> Tuple[float, float, float]:
    brain.eval()  # Ensure the model is in evaluation mode

    losses: dict[str, List[torch.Tensor]]
    losses = {"total": [], "classification": [], "reconstruction": []}

    with torch.no_grad():  # Disable gradient calculation
        for batch in testloader:
            loss, class_loss, recon_loss = calculate_loss(
                device, brain, recon_weight, recon_objective, class_objective, batch
            )

            losses["total"].append(loss)
            losses["classification"].append(class_loss)
            losses["reconstruction"].append(recon_loss)

    avg_loss = torch.mean(torch.stack(losses["total"])).item()
    avg_class_loss = torch.mean(torch.stack(losses["classification"])).item()
    avg_recon_loss = torch.mean(torch.stack(losses["reconstruction"])).item()
    return avg_loss, avg_class_loss, avg_recon_loss
