### Imports ###

import json
import multiprocessing as mp
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from retinal_rl.models.brain import Brain


def single_run(
    device: torch.device,
    brain: Brain,
    num_epochs: int,
    recon_weight: float,
    dataset: Dataset[Tensor],
    split_ratio: float = 0.8,
) -> Tuple[Brain, dict[str, List[float]]]:
    """Performs a single run with a train/validation split."""
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, validation_set = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    validationloader = DataLoader(validation_set, batch_size=64, shuffle=False)

    class_objective = nn.CrossEntropyLoss()
    recon_objective = nn.MSELoss()
    optimizer = optim.Adam(brain.parameters(), lr=0.001)

    history: dict[str, List[float]]
    history = {
        "train_total": [],
        "train_classification": [],
        "train_reconstruction": [],
        "validation_total": [],
        "validation_classification": [],
        "validation_reconstruction": [],
    }

    for _ in range(num_epochs):
        train_loss, train_recon_loss, train_class_loss = _train_epoch(
            device,
            brain,
            optimizer,
            recon_weight,
            recon_objective,
            class_objective,
            trainloader,
        )
        val_loss, val_recon_loss, val_class_loss = _validate_model(
            device,
            brain,
            recon_weight,
            recon_objective,
            class_objective,
            validationloader,
        )

        history["train_total"].append(train_loss)
        history["train_classification"].append(train_class_loss)
        history["train_reconstruction"].append(train_recon_loss)
        history["validation_total"].append(val_loss)
        history["validation_classification"].append(val_class_loss)
        history["validation_reconstruction"].append(val_recon_loss)

    return brain, history


def cross_validate(
    device: torch.device,
    brain_factory: Callable[[], Brain],
    num_folds: int,
    num_epochs: int,
    recon_weight: float,
    dataset: Dataset[Tensor],
) -> Tuple[List[nn.Module], List[dict[str, List[float]]]]:
    fold_size = len(dataset) // num_folds
    folds = random_split(
        dataset,
        [fold_size] * (num_folds - 1) + [len(dataset) - fold_size * (num_folds - 1)],
    )

    mp_args = []

    for fold in range(num_folds):
        train_subsets = [xs for i, xs in enumerate(folds) if i != fold]
        trainset: Dataset[Tensor]
        trainset = ConcatDataset(train_subsets)
        valset = folds[fold]
        mp_args.append(
            (
                fold,
                device.type,
                brain_factory,
                trainset,
                valset,
                num_epochs,
                recon_weight,
            )
        )

    with mp.Pool(processes=num_folds) as pool:
        results = pool.map(_run_fold, mp_args)

    brains, histories = zip(*[(result[1], result[2]) for result in results])

    return list(brains), list(histories)


def save_results(
    brains: List[nn.Module],
    histories: List[dict[str, List[float]]],
) -> None:
    # Save histories as a single JSON file
    histories_file_path = "histories.json"
    with open(histories_file_path, "w") as f:
        json.dump(histories, f)

    # Save each model
    for fold, model in enumerate(brains):
        model_file_path = f"model_{fold + 1}.pt"
        torch.save(model.state_dict(), model_file_path)


def _calculate_loss(
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


def _train_epoch(
    device: torch.device,
    brain: Brain,
    optimizer: Optimizer,
    recon_weight: float,
    recon_objective: nn.Module,
    class_objective: nn.Module,
    trainloader: DataLoader[Tensor],
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

        loss, class_loss, recon_loss = _calculate_loss(
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


def _validate_model(
    device: torch.device,
    brain: Brain,
    recon_weight: float,
    recon_objective: torch.nn.Module,
    class_objective: torch.nn.Module,
    validationloader: DataLoader[Tensor],
) -> Tuple[float, float, float]:
    brain.eval()  # Ensure the model is in evaluation mode

    losses: dict[str, List[torch.Tensor]]
    losses = {"total": [], "classification": [], "reconstruction": []}

    with torch.no_grad():  # Disable gradient calculation
        for batch in validationloader:
            loss, class_loss, recon_loss = _calculate_loss(
                device, brain, recon_weight, recon_objective, class_objective, batch
            )

            losses["total"].append(loss)
            losses["classification"].append(class_loss)
            losses["reconstruction"].append(recon_loss)

    avg_loss = torch.mean(torch.stack(losses["total"])).item()
    avg_class_loss = torch.mean(torch.stack(losses["classification"])).item()
    avg_recon_loss = torch.mean(torch.stack(losses["reconstruction"])).item()
    return avg_loss, avg_class_loss, avg_recon_loss


def _run_fold(
    args: Tuple[
        int,
        str,
        Callable[[], Brain],
        Dataset[Tensor],
        Dataset[Tensor],
        int,
        float,
    ],
) -> Tuple[int, Brain, dict[str, List[float]]]:
    (
        fold,
        device_str,
        brain_factory,
        train_set,
        validation_set,
        num_epochs,
        recon_weight,
    ) = args
    device = torch.device(device_str)
    return _train_fold(
        fold,
        device,
        brain_factory,
        train_set,
        validation_set,
        num_epochs,
        recon_weight,
    )


def _train_fold(
    fold: int,
    device: torch.device,
    brain_factory: Callable[[], Brain],
    train_set: Dataset[Tensor],
    validation_set: Dataset[Tensor],
    num_epochs: int,
    recon_weight: float,
) -> Tuple[int, Brain, dict[str, List[float]]]:
    brain = brain_factory()
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    validationloader = DataLoader(validation_set, batch_size=64, shuffle=False)

    class_objective = nn.CrossEntropyLoss()
    recon_objective = nn.MSELoss()
    optimizer = optim.Adam(brain.parameters(), lr=0.001)

    history: dict[str, List[float]]
    history = {
        "train_total": [],
        "train_classification": [],
        "train_reconstruction": [],
        "validation_total": [],
        "validation_classification": [],
        "validation_reconstruction": [],
    }

    for _ in range(num_epochs):
        train_loss, train_recon_loss, train_class_loss = _train_epoch(
            device,
            brain,
            optimizer,
            recon_weight,
            recon_objective,
            class_objective,
            trainloader,
        )
        val_loss, val_recon_loss, val_class_loss = _validate_model(
            device,
            brain,
            recon_weight,
            recon_objective,
            class_objective,
            validationloader,
        )

        history["train_total"].append(train_loss)
        history["train_classification"].append(train_class_loss)
        history["train_reconstruction"].append(train_recon_loss)
        history["validation_total"].append(val_loss)
        history["validation_classification"].append(val_class_loss)
        history["validation_reconstruction"].append(val_recon_loss)

    return fold, brain, history
