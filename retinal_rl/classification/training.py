"""Training module for the Brain model.

This module contains functions for running training epochs, processing datasets,
and calculating losses for the Brain model. It works in conjunction with the
Brain and BrainOptimizer classes to perform model training and evaluation.
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from retinal_rl.models.brain import Brain
from retinal_rl.models.optimizer import BrainOptimizer

logger = logging.getLogger(__name__)


def run_epoch(
    device: torch.device,
    brain: Brain,
    brain_optimizer: BrainOptimizer,
    history: Dict[str, List[float]],
    epoch: int,
    trainloader: DataLoader[Tuple[Tensor, int]],
    testloader: DataLoader[Tuple[Tensor, int]],
) -> Tuple[Brain, Dict[str, List[float]]]:
    """Perform a single training epoch and evaluation.

    This function runs the model through one complete pass of the training data
    and then evaluates it on the test data. It updates the training history with
    the results.

    Args:
    ----
        device (torch.device): The device to run the computations on.
        brain (Brain): The Brain model to train and evaluate.
        brain_optimizer (BrainOptimizer): The optimizer for updating the model parameters.
        history (Dict[str, List[float]]): A dictionary to store the training history.
        epoch (int): The current epoch number.
        trainloader (DataLoader): DataLoader for the training dataset.
        testloader (DataLoader): DataLoader for the test dataset.

    Returns:
    -------
        Tuple[Brain, Dict[str, List[float]]]: The updated Brain model and the updated history.

    """
    train_losses = process_dataset(
        device, brain, brain_optimizer, epoch, trainloader, is_training=True
    )
    test_losses = process_dataset(
        device, brain, brain_optimizer, epoch, testloader, is_training=False
    )

    # Update history
    logger.info(f"Epoch {epoch} training performance:")
    for key, value in train_losses.items():
        logger.info(f"{key}: {value:.4f}")
        history.setdefault(f"train_{key}", []).append(value)
    for key, value in test_losses.items():
        history.setdefault(f"test_{key}", []).append(value)

    return brain, history


def process_dataset(
    device: torch.device,
    brain: Brain,
    brain_optimizer: BrainOptimizer,
    epoch: int,
    dataloader: DataLoader[Tuple[Tensor, int]],
    is_training: bool,
) -> Dict[str, float]:
    """Process a dataset (train or test) and return average losses.

    This function runs the model on all batches in the given dataset. If in training mode,
    it also performs optimization steps.

    Args:
    ----
        device (torch.device): The device to run the computations on.
        brain (Brain): The Brain model to process the data.
        brain_optimizer (BrainOptimizer): The optimizer for updating the model parameters.
        epoch (int): The current epoch number.
        dataloader (DataLoader): The DataLoader containing the dataset to process.
        is_training (bool): Whether to perform optimization (True) or just evaluate (False).

    Returns:
    -------
        Dict[str, float]: A dictionary of average losses for the processed dataset.

    """
    total_losses: Dict[str, float] = {}
    steps = 0

    for batch in dataloader:
        context = get_context(device, brain, epoch, batch)

        if is_training:
            brain.train()
            losses, obj_dict = brain_optimizer.optimize(context)
        else:
            brain.eval()
            losses, obj_dict = brain_optimizer.compute_losses(context)

        # Accumulate losses and objectives
        for key, value in losses.items():
            total_losses[key] = total_losses.get(key, 0.0) + value
        for key, value in obj_dict.items():
            total_losses[key] = total_losses.get(key, 0.0) + value

        steps += 1

    # Calculate average losses
    return {key: value / steps for key, value in total_losses.items()}


def get_context(
    device: torch.device,
    brain: Brain,
    epoch: int,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> Dict[str, Any]:
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

    return {
        "inputs": inputs,
        "classes": classes,
        "responses": responses,
        "parameters": brain.parameters(),
        "epoch": epoch,
    }
