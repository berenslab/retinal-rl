"""Module for managing optimization of complex neural network models with multiple circuits."""

import logging
from typing import Dict, Generic, List

import torch
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT, Loss

logger = logging.getLogger(__name__)


class Objective(Generic[ContextT]):
    """Manages multiple loss functions that target NeuralCircuits in a Brain.

    This class handles the initialization, state management, and optimization steps
    for multiple optimizers, each associated with specific circuits and objectives.


    Attributes
    ----------
        brain (Brain): The neural network model being optimized.
        losses (OrderedDict[str, Optimizer]): Instantiated optimizers, sorted based on connectome.

    """

    def __init__(self, brain: Brain, optimizer: Optimizer, losses: List[Loss[ContextT]]):
        """Initialize the BrainOptimizer.

        Args:
        ----
            brain (Brain): The neural network model to optimize.
            optimizer (Optimizer): The optimizer to use for training.
            losses (List[Loss[ContextT]]): A list of loss functions to optimize.

        Raises:
        ------
            ValueError: If a specified circuit is not found in the brain.

        """
        self.device = next(brain.parameters()).device
        self.optimizer = optimizer
        self.losses: List[Loss[ContextT]] = losses
        self.params: List[List[Parameter]] = []

        for loss in self.losses:
            # Collect parameters from target circuits
            params: List[Parameter] = []
            for circuit_name in loss.target_circuits:
                if circuit_name in brain.circuits:
                    params.extend(brain.circuits[circuit_name].parameters())

            self.params.append(params)

    def backward(self, context: ContextT) -> Dict[str, float]:
        loss_dict: Dict[str, float] = {}

        retain_graph = True

        for i, (loss, params) in enumerate(zip(self.losses, self.params)):
            # Compute losses
            name = loss.key_name
            weights = loss.weights
            value = loss(context)
            loss_dict[name] = value.item()

            # Skip training if the optimizer is not at a training epoch
            if not loss.is_training_epoch(context.epoch):
                continue

            # Set retain_graph to True for all but the last optimizer
            retain_graph = i < len(self.losses) - 1

            # Compute gradients
            grads = torch.autograd.grad(
                value, params, create_graph=False, retain_graph=retain_graph
            )

            # Manually update parameters
            with torch.no_grad():
                for param, weight, grad in zip(params, weights, grads):
                    if param.grad is None:
                        param.grad = weight * grad
                    else:
                        param.grad += weight * grad

        # Perform optimization step
        return loss_dict

    def num_epochs(self) -> int:
        """Get the maximum number of epochs over all optimizers.

        Returns
        -------
            int: The maximum number of epochs across all losses.

        """
        return max([loss.max_epoch for loss in self.losses])
