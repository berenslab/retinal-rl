"""Training module for the Brain model.

This module contains functions for running training epochs, processing datasets,
and calculating losses for the Brain model. It works in conjunction with the
Brain and BrainOptimizer classes to perform model training and evaluation.
"""

import logging
import time
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from retinal_rl.classification.loss import (
    ClassificationContext,
    get_classification_context,
)
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import Objective

logger = logging.getLogger(__name__)


def run_epoch(
    device: torch.device,
    brain: Brain,
    objective: Objective[ClassificationContext],
    optimizer: Optimizer,
    history: Dict[str, List[float]],
    epoch: int,
    trainloader: DataLoader[Tuple[Tensor, Tensor, int]],
    testloader: DataLoader[Tuple[Tensor, Tensor, int]],
    gpu_transforms=None,
    batch_transforms=None,
) -> Tuple[Brain, Dict[str, List[float]]]:
    """Perform a single training epoch and evaluation.

    This function runs the model through one complete pass of the training data
    and then evaluates it on the test data. It updates the training history with
    the results.

    Args:
    ----
        device (torch.device): The device to run the computations on.
        brain (Brain): The Brain model to train and evaluate.
        objective (Objective): The objective object specifying the training objectives.
        optimizer (Optimizer): The optimizer for updating the model parameters.
        history (Dict[str, List[float]]): A dictionary to store the training history.
        epoch (int): The current epoch number.
        trainloader (DataLoader): DataLoader for the training dataset.
        testloader (DataLoader): DataLoader for the test dataset.

    Returns:
    -------
        Tuple[Brain, Dict[str, List[float]]]: The updated Brain model and the updated history.

    """
    print(f"  run_epoch() breakdown for epoch {epoch}:")
    
    train_start = time.time()
    train_losses = process_dataset(
        device, brain, objective, optimizer, epoch, trainloader, is_training=True, gpu_transforms=gpu_transforms, batch_transforms=batch_transforms
    )
    train_time = time.time() - train_start
    print(f"    Training: {train_time:.3f}s")
    
    test_start = time.time()
    test_losses = process_dataset(
        device, brain, objective, optimizer, epoch, testloader, is_training=False, gpu_transforms=gpu_transforms, batch_transforms=batch_transforms
    )
    test_time = time.time() - test_start
    print(f"    Testing: {test_time:.3f}s")

    # Update history
    history_start = time.time()
    logger.info(f"Epoch {epoch} training performance:")
    for key, value in train_losses.items():
        logger.info(f"{key}: {value:.4f}")
        history.setdefault(f"train_{key}", []).append(value)
    for key, value in test_losses.items():
        history.setdefault(f"test_{key}", []).append(value)
    history_time = time.time() - history_start
    print(f"    History update: {history_time:.3f}s")
    
    total_run_epoch = train_time + test_time + history_time
    print(f"    run_epoch total: {total_run_epoch:.3f}s")

    return brain, history


def process_dataset(
    device: torch.device,
    brain: Brain,
    objective: Objective[ClassificationContext],
    optimizer: Optimizer,
    epoch: int,
    dataloader: DataLoader[Tuple[Tensor, Tensor, int]],
    is_training: bool,
    gpu_transforms=None,
    batch_transforms=None,
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

    batch_times = []
    context_times = []
    forward_times = []
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Time context creation
        context_start = time.time()
        context = get_classification_context(device, brain, batch, epoch, gpu_transforms, batch_transforms)
        context_time = time.time() - context_start
        context_times.append(context_time)

        if is_training:
            brain.train()
            
            # Time loss computation
            loss_start = time.time()
            losses = objective.backward(context)
            loss_time = time.time() - loss_start
            forward_times.append(loss_time)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        else:
            with torch.no_grad():
                brain.eval()
                losses: Dict[str, float] = {}
                for stat in objective.logging_statistics:
                    losses[stat.key_name] = stat(context).item()
                for loss in objective.losses:
                    losses[loss.key_name] = loss(context).item()
            loss_time = 0  # No loss timing for eval

        # Accumulate losses and objectives
        for key, value in losses.items():
            total_losses[key] = total_losses.get(key, 0.0) + value

        steps += 1
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Print timing for first few batches
        if batch_idx < 3:
            print(f"    Batch {batch_idx}: total={batch_time:.3f}s, context={context_time:.3f}s, loss={loss_time if is_training else 0:.3f}s")

    # Print timing summary
    if batch_times:
        avg_batch = sum(batch_times) / len(batch_times)
        avg_context = sum(context_times) / len(context_times) 
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        print(f"  Timing summary: avg_batch={avg_batch:.3f}s, avg_context={avg_context:.3f}s, avg_loss={avg_forward:.3f}s")

    # Calculate average losses
    return {key: value / steps for key, value in total_losses.items()}
