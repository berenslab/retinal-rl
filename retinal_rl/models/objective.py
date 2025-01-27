"""Module for managing optimization of complex neural network models with multiple circuits."""

import logging
from typing import Dict, Generic, List, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT, LoggingStatistic, Loss

logger = logging.getLogger(__name__)


class Objective(Generic[ContextT]):
    def __init__(
        self,
        brain: Brain,
        losses: List[Loss[ContextT]],
        logging_statistics: Optional[List[LoggingStatistic[ContextT]]] = None,
    ):
        if logging_statistics is None:
            logging_statistics = []

        for loss in losses:
            assert isinstance(loss, Loss), "losses need to subclass Loss"

        for stat in logging_statistics:
            assert isinstance(
                stat, LoggingStatistic
            ), "logging_statistics need to subclass LoggingStatistic"

        self.device = next(brain.parameters()).device
        self.losses = losses
        self.logging_statistics = logging_statistics
        self.brain: Brain = brain

        # Build a dictionary of weighted parameters for each loss
        # TODO: If the parameters() list of a neural circuit changes dynamically, this will break

    def backward(self, context: ContextT) -> Dict[str, float]:
        loss_dict: Dict[str, float] = {}

        retain_graph = True

        for i, stat in enumerate(self.logging_statistics):
            loss_dict[stat.key_name] = stat(context).item()

        for i, loss in enumerate(self.losses):
            name = loss.key_name
            value = loss(context)
            loss_dict[name] = value.item()

            # Compute losses
            weights, params = self._weighted_params(loss)
            if not loss.is_training_epoch(context.epoch) or not params:
                continue

            # Set retain_graph to True for all but the last optimizer
            retain_graph = i < len(self.losses) - 1

            # Compute gradients
            grads = torch.autograd.grad(
                value,
                params,
                create_graph=False,
                retain_graph=retain_graph,
                allow_unused=True, # TODO: check if this is actually the desired behavior. Needed to be added for when eg a head is not used for loss computation, but we have target all
            )

            # Manually update parameters
            with torch.no_grad():
                for param, weight, grad in zip(params, weights, grads):
                    if grad is not None: # TODO: also only added as allow_unused above
                        if param.grad is None:
                            param.grad = weight * grad
                        else:
                            param.grad += weight * grad

        # Perform optimization step
        return loss_dict

    def _weighted_params(
        self, loss: Loss[ContextT]
    ) -> Tuple[List[float], List[Parameter]]:
        _targets = loss.target_circuits
        _weights = loss.weights

        if "__all__" in _targets:
            _targets = self.brain.circuits.keys()
            if len(_weights) == 1:
                _weights = [_weights[0] for _ in range(len(_targets))]
            assert len(_weights) == len(_targets)

        weights: List[float] = []
        params: List[Parameter] = []
        for weight, circuit_name in zip(_weights, _targets):
            if circuit_name in self.brain.circuits:
                params0 = list(self.brain.circuits[circuit_name].parameters())
                weights += [weight] * len(params0)
                params += params0

        return weights, params
