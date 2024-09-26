"""Debug module for comparing gradient computations in the Brain model training process."""

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from retinal_rl.classification.objective import (
    ClassificationContext,
    get_classification_context,
)
from retinal_rl.dataset import Imageset
from retinal_rl.models.brain import Brain
from retinal_rl.models.optimizer import BrainOptimizer

logger = logging.getLogger(__name__)


def get_optimizer_params(optimizer: Optimizer) -> Set[torch.Tensor]:
    """Get the parameters of an optimizer."""
    params: Set[torch.Tensor] = set()
    for param_group in optimizer.param_groups:
        params.update(set(param_group["params"]))
    return params


def check_parameter_overlap(
    brain_optimizer: BrainOptimizer[ClassificationContext],
) -> None:
    """Check for parameter overlap between optimizers."""
    param_sets: Dict[str, Set[torch.Tensor]] = {}

    for optimizer_name, optimizer in brain_optimizer.optimizers.items():
        param_sets[optimizer_name] = get_optimizer_params(optimizer)

    overlaps: Dict[Tuple[str, str], Set[torch.Tensor]] = {}
    for name1, params1 in param_sets.items():
        for name2, params2 in param_sets.items():
            if name1 < name2:  # Avoid duplicate comparisons
                shared_params = params1.intersection(params2)
                if shared_params:
                    overlaps[(name1, name2)] = shared_params

    if overlaps:
        warning_msg = "Parameter overlap detected between optimizers:"
        for (opt1, opt2), shared in overlaps.items():
            warning_msg += f"\n  - {opt1} and {opt2}: {len(shared)} shared parameters"
        logger.critical(warning_msg)
    else:
        logger.info("No optimizer parameter overlap detected.")


def compare_gradient_computation(
    device: torch.device,
    brain: Brain,
    optimizer: BrainOptimizer[ClassificationContext],
    dataset: Imageset,
) -> Tuple[bool, Dict[str, Optional[float]]]:
    """Compare gradient computations between efficient and ground truth methods.

    Args:
    ----
        device (torch.device): The device to run computations on.
        brain (Brain): The Brain model.
        optimizer (BrainOptimizer[ClassificationContext]): The optimizer for the Brain model.
        dataloader (DataLoader): DataLoader for the dataset.
        num_batches (int): Number of batches to process for comparison.

    Returns:
    -------
        Tuple[bool, Dict[str, List[float]]]: A tuple containing:
            - A boolean indicating whether all gradients match within tolerance.
            - A dictionary of discrepancies for each batch.

    """
    all_match = True
    discrepancies: Dict[str, Optional[float]] = {}
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for batch_idx, batch in enumerate(dataloader):
        context = get_classification_context(device, brain, batch, 0)

        efficient_grads = compute_efficient_gradients(optimizer, context)
        ground_truth_grads = compute_ground_truth_gradients(
            optimizer, brain, device, batch
        )

        match, discrepancies = compare_gradients(efficient_grads, ground_truth_grads)

        all_match = all_match and match

        if not match:
            print(f"Discrepancies found in batch {batch_idx}")

        break

    return all_match, discrepancies


def compute_efficient_gradients(
    brain_optimizer: BrainOptimizer[ClassificationContext], context: ClassificationContext
) -> Dict[str, List[Tensor | None]]:
    """Compute gradients using the efficient method."""
    grads: Dict[str, List[Tensor | None]] = {
        name: [] for name in brain_optimizer.optimizers.keys()
    }

    retain_graph = True
    for i, (name, opt) in enumerate(brain_optimizer.optimizers.items()):
        opt.zero_grad()
        if i == len(brain_optimizer.optimizers) - 1:
            retain_graph = False
        loss, _ = brain_optimizer.compute_loss(name, context)
        loss.backward(retain_graph=retain_graph)
        grads[name] = [
            param.grad.clone() if param.grad is not None else None
            for param in get_optimizer_params(opt)
        ]

    return grads


def compute_ground_truth_gradients(
    brain_optimizer: BrainOptimizer[ClassificationContext],
    brain: Brain,
    device: torch.device,
    batch: Tuple[Tensor, Tensor],
) -> Dict[str, List[Tensor | None]]:
    """Compute gradients using the ground truth method, rebuilding the context for each optimizer."""
    grads: Dict[str, List[Tensor | None]] = {
        name: [] for name in brain_optimizer.optimizers.keys()
    }

    for opt in brain_optimizer.optimizers.values():
        opt.zero_grad()

    for name, opt in brain_optimizer.optimizers.items():
        # Use get_classification_context to ensure a fresh computational graph
        context = get_classification_context(device, brain, 0, batch)
        loss, _ = brain_optimizer.compute_loss(name, context)
        loss.backward()

        grads[name] = [
            param.grad.clone() if param.grad is not None else None
            for param in get_optimizer_params(opt)
        ]

    return grads


def compare_gradients(
    efficient_grads: Dict[str, List[Tensor | None]],
    ground_truth_grads: Dict[str, List[Tensor | None]],
) -> Tuple[bool, Dict[str, Optional[float]]]:
    """Compare gradients and return match status and discrepancies."""
    match = True
    discrepancies: Dict[str, Optional[float]] = {}

    for name in efficient_grads.keys():
        for i, (eff_grad, gt_grad) in enumerate(
            zip(efficient_grads[name], ground_truth_grads[name])
        ):
            if eff_grad is None and gt_grad is None:
                continue
            if eff_grad is None or gt_grad is None:
                match = False
                discrepancies[f"{name}_param_{i}_grad_mismatch"] = None
                continue
            diff = torch.max(torch.abs(eff_grad - gt_grad)).item()
            if diff > 1e-6:  # Tolerance threshold
                match = False
                discrepancies[f"{name}_param_{i}_grad_diff"] = diff

    return match, discrepancies
