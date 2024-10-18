"""Module for managing optimization of complex neural network models with multiple circuits."""

import logging
from typing import Dict, Generic, List, Tuple

import torch
from torch.nn.parameter import Parameter

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT, Loss

logger = logging.getLogger(__name__)


class Objective(Generic[ContextT]):
    def __init__(self, brain: Brain, losses: List[Loss[ContextT]]):
        self.device = next(brain.parameters()).device
        self.losses: List[Loss[ContextT]] = losses
        self.brain: Brain = brain

        # Build a dictionary of weighted parameters for each loss
        # TODO: If the parameters() list of a neural circuit changes dynamically, this will break

    def backward(self, context: ContextT) -> Dict[str, float]:
        loss_dict: Dict[str, float] = {}

        retain_graph = True

        for i, loss in enumerate(self.losses):
            # Compute losses
            weights, params = self._weighted_params(loss)
            name = loss.key_name
            value = loss(context)
            loss_dict[name] = value.item()
            if not loss.is_training_epoch(context.epoch) or not params:
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

    def _weighted_params(
        self, loss: Loss[ContextT]
    ) -> Tuple[List[float], List[Parameter]]:
        weights: List[float] = []
        params: List[Parameter] = []
        for weight, circuit_name in zip(loss.weights, loss.target_circuits):
            if circuit_name in self.brain.circuits:
                params0 = list(self.brain.circuits[circuit_name].parameters())
                weights += [weight] * len(params0)
                params += params0

        return weights, params
