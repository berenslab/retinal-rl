### Imports ###

import os
import json
import multiprocessing as mp

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torch.optim.optimizer import Optimizer

from retinal_classification.models import ConvAutoencoder, AutoencodingClassifier


def calculate_loss(
    inputs: torch.Tensor,
    true_classes: torch.Tensor,
    predicted_classes: torch.Tensor,
    reconstructions: torch.Tensor,
    class_objective: nn.Module,
    recon_objective: nn.Module,
    recon_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    class_loss = class_objective(predicted_classes, true_classes)
    recon_loss = recon_objective(inputs, reconstructions)

    class_weight = 1 - recon_weight
    loss = class_weight * class_loss + recon_weight * recon_loss

    return loss, class_loss, recon_loss


def train_epoch(
    device: torch.device,
    model: AutoencodingClassifier,
    optimizer: Optimizer,
    recon_weight: float,
    recon_objective: nn.Module,
    class_objective: nn.Module,
    trainloader: DataLoader,
) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch.

    Returns:
        Tuple[float, float,float]: A tuple containing the average loss,
        reconstruction loss, and classification loss for the epoch.
    """

    losses: dict[str, List[torch.Tensor]]
    losses = {"total": [], "classification": [], "reconstruction": []}

    for batch in trainloader:

        inputs, classes = batch
        inputs, classes = inputs.to(device), classes.to(device)
        optimizer.zero_grad()
        predicted_classes, reconstructions = model(inputs)

        loss, class_loss, recon_loss = calculate_loss(
            inputs,
            classes,
            predicted_classes,
            reconstructions,
            class_objective,
            recon_objective,
            recon_weight,
        )

        loss.backward()
        optimizer.step()

        losses["total"].append(loss)
        losses["classification"].append(class_loss)
        losses["reconstruction"].append(recon_loss)

    avg_loss = torch.mean(torch.stack(losses["total"])).item()
    avg_class_loss = torch.mean(torch.stack(losses["classification"])).item()
    avg_recon_loss = torch.mean(torch.stack(losses["reconstruction"])).item()
    return avg_loss, avg_class_loss, avg_recon_loss


def validate_model(
    device: torch.device,
    model: AutoencodingClassifier,
    recon_weight: float,
    recon_objective: torch.nn.Module,
    class_objective: torch.nn.Module,
    validationloader: DataLoader,
) -> Tuple[float, float, float]:
    model.eval()  # Ensure the model is in evaluation mode

    losses: dict[str, List[torch.Tensor]]
    losses = {"total": [], "classification": [], "reconstruction": []}

    with torch.no_grad():  # Disable gradient calculation
        for batch in validationloader:
            inputs, classes = batch
            inputs, classes = inputs.to(device, non_blocking=True), classes.to(
                device, non_blocking=True
            )
            outputs, predicted_classes = model(inputs)
            loss, class_loss, recon_loss = calculate_loss(
                outputs,
                inputs,
                predicted_classes,
                classes,
                recon_objective,
                class_objective,
                recon_weight,
            )
            losses["total"].append(loss)
            losses["classification"].append(class_loss)
            losses["reconstruction"].append(recon_loss)

    avg_loss = torch.mean(torch.stack(losses["total"])).item()
    avg_class_loss = torch.mean(torch.stack(losses["classification"])).item()
    avg_recon_loss = torch.mean(torch.stack(losses["reconstruction"])).item()
    return avg_loss, avg_class_loss, avg_recon_loss


def train_fold(
    fold: int,
    device: torch.device,
    model_class: type,
    train_set: Dataset,
    validation_set: Dataset,
    num_epochs: int,
    recon_weight: float,
) -> Tuple[int, AutoencodingClassifier, dict[str, List[float]]]:
    """
    Trains the model for one fold of the cross-validation.
    """

    model = model_class().to(device)
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    validationloader = DataLoader(validation_set, batch_size=64, shuffle=False)

    class_objective = nn.CrossEntropyLoss()
    recon_objective = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        train_loss, train_recon_loss, train_class_loss = train_epoch(
            device,
            model,
            optimizer,
            recon_weight,
            recon_objective,
            class_objective,
            trainloader,
        )
        val_loss, val_recon_loss, val_class_loss = validate_model(
            device,
            model,
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

    return fold, model, history


def run_fold(
    args: Tuple[int, str, type, Dataset, Dataset, int, float]
) -> Tuple[int, AutoencodingClassifier, dict[str, List[float]]]:
    (
        fold,
        device_str,
        model_class,
        train_set,
        validation_set,
        num_epochs,
        recon_weight,
    ) = args
    device = torch.device(device_str)
    return train_fold(
        fold, device, model_class, train_set, validation_set, num_epochs, recon_weight
    )


def cross_validate(
    device: torch.device,
    num_folds: int,
    num_epochs: int,
    recon_weight: float,
    dataset: Dataset,
) -> Tuple[List[nn.Module], List[dict[str, List[float]]]]:

    fold_size = len(dataset) // num_folds
    folds = random_split(
        dataset,
        [fold_size] * (num_folds - 1) + [len(dataset) - fold_size * (num_folds - 1)],
    )

    model_class = ConvAutoencoder

    mp_args = []

    for fold in range(num_folds):
        train_subsets = [xs for i, xs in enumerate(folds) if i != fold]
        trainset: Dataset
        trainset = ConcatDataset(train_subsets)
        valset = folds[fold]
        mp_args.append(
            (fold, device.type, model_class, trainset, valset, num_epochs, recon_weight)
        )

    with mp.Pool(processes=num_folds) as pool:
        results = pool.map(run_fold, mp_args)

    models, histories = zip(*[(result[1], result[2]) for result in results])

    return list(models), list(histories)


def save_results(
    models: List[nn.Module],
    histories: List[dict],
    results_folder: str,
    recon_weight: float,
) -> None:
    results_folder = os.path.join(results_folder, f"lambda_{recon_weight}")
    os.makedirs(results_folder, exist_ok=True)

    # Save histories as a single JSON file
    histories_file_path = os.path.join(results_folder, "histories.json")
    with open(histories_file_path, "w") as f:
        json.dump(histories, f)

    # Save each model
    for fold, model in enumerate(models):
        model_file_path = os.path.join(results_folder, f"model_{fold + 1}.pt")
        torch.save(model.state_dict(), model_file_path)
