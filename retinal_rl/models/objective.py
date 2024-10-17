"""Module for managing optimization of complex neural network models with multiple circuits."""

import logging
from typing import Dict, Generic, List

import torch
from torch.nn.parameter import Parameter

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT, Loss

logger = logging.getLogger(__name__)


class Objective(Generic[ContextT]):
    def __init__(self, brain: Brain, losses: List[Loss[ContextT]]):
        self.device = next(brain.parameters()).device
        self.losses: List[Loss[ContextT]] = losses
        self.paramss: List[List[Parameter]] = []

        for loss in self.losses:
            # Collect parameters from target circuits
            params: List[Parameter] = []
            for circuit_name in loss.target_circuits:
                if circuit_name in brain.circuits:
                    params.extend(brain.circuits[circuit_name].parameters())
            self.paramss.append(params)

    def backward(self, context: ContextT) -> Dict[str, float]:
        loss_dict: Dict[str, float] = {}

        retain_graph = True

        for i, (loss, params) in enumerate(zip(self.losses, self.paramss)):
            # Compute losses
            name = loss.key_name
            weights = loss.weights
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
