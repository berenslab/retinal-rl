"""Module for managing optimization of complex neural network models with multiple circuits."""

import logging
from typing import Dict, Generic, List, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.parameter import Parameter

from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT, Loss

logger = logging.getLogger(__name__)


class Goal(Generic[ContextT]):
    """Manages multiple optimizers that target NeuralCircuits in a Brain.

    This class handles the initialization, state management, and optimization steps
    for multiple optimizers, each associated with specific circuits and objectives.


    Attributes
    ----------
        brain (Brain): The neural network model being optimized.
        losses (OrderedDict[str, Optimizer]): Instantiated optimizers, sorted based on connectome.
        objectives (Dict[str, List[WeightedLoss]]): Losses for each optimizer.
        target_circuits (Dict[str, List[str]]): Target circuits for each optimizer.
        min_epochs (Dict[str, int]): Minimum epochs for each optimizer.
        max_epochs (Dict[str, int]): Maximum epochs for each optimizer.

    """

    def __init__(self, brain: Brain, objective_configs: Dict[str, DictConfig]):
        """Initialize the BrainOptimizer.

        Args:
        ----
            brain (Brain): The neural network model to optimize.
            optimizer (Optimizer): The optimizer to use for training.
            objective_configs (Dict[str, DictConfig]): Configuration for each optimizer.
                Each config should specify target_circuits, optimizer settings, and objectives.

        Raises:
        ------
            ValueError: If a specified circuit is not found in the brain.

        """
        self.device = next(brain.parameters()).device
        self.losses: Dict[str, List[Loss[ContextT]]] = {}
        self.target_circuits: Dict[str, List[str]] = {}
        self.min_epochs: Dict[str, int] = {}
        self.max_epochs: Dict[str, int] = {}
        self.params: Dict[str, List[Parameter]] = {}

        for objective, config in objective_configs.items():
            # Collect parameters from target circuits
            params = []
            self.min_epochs[objective] = config.get("min_epoch", 0)
            self.max_epochs[objective] = config.get("max_epoch", -1)
            self.target_circuits[objective] = config.target_circuits
            if not set(config.target_circuits).issubset(brain.connectome.nodes):
                logger.warning(
                    f"Some target circuits for objective: {objective} are not in the brain's connectome"
                )
            for circuit_name in config.target_circuits:
                if circuit_name in brain.circuits:
                    params.extend(brain.circuits[circuit_name].parameters())

            self.params[objective] = params

            # Initialize objectives
            self.losses[objective] = [
                instantiate(obj_config) for obj_config in config.losses
            ]
            logger.info(
                f"Initalized objective: {objective}, with losses: {[obj.key_name for obj in self.losses[objective]]}, and target circuits: {[circuit_name for circuit_name in self.target_circuits]}"
            )

    def evaluate_objective(
        self, objective: str, context: ContextT
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss for a specific objective.

        Args:
        ----
            objective (str): Name of the objective.
            context (ContextT]): Context information for computing objectives.

        Returns:
        -------
            Tuple[torch.Tensor, Dict[str, float]]: A tuple containing the total loss
                and a dictionary of raw loss values for each objective.

        """
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict: Dict[str, float] = {}
        for loss in self.losses[objective]:
            weighted_loss, raw_loss = loss(context)
            total_loss += weighted_loss
            loss_dict[loss.key_name] = raw_loss.item()
        return total_loss, loss_dict

    def evaluate_objectives(
        self, context: ContextT
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute all objectives without computing gradients.

        This method is useful for evaluation purposes.

        Args:
        ----
            context (Dict[str, Any]): Context information for computing objectives.

        Returns:
        -------
            Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries
                of total objectives and raw loss values for each objective.

        """
        objectives: Dict[str, float] = {}
        loss_dict: Dict[str, float] = {}
        for objective in self.losses.keys():
            loss, sub_obj_dict = self.evaluate_objective(objective, context)
            objectives[f"{objective}_objective"] = loss.item()
            loss_dict.update(sub_obj_dict)
        return objectives, loss_dict

    def _is_training_epoch(self, name: str, epoch: int) -> bool:
        """Check if the objective should currently be pursued.

        Args:
        ----
            name (str): Name of the optimizer.
            epoch (int): Current epoch number.

        Returns:
        -------
            bool: True if the objective should continue training, False otherwise.

        """
        if epoch < self.min_epochs[name]:
            return False
        if self.max_epochs[name] < 0:
            return True
        return epoch < self.max_epochs[name]

    def backward(self, context: ContextT) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute a backward pass over the brain with respect to all objectives.

        This method computes losses, performs backpropagation, and updates parameters
        for all NeuralCircuits.

        Args:
        ----
            context (ContextT): Context information for computing objectives.

        Returns:
        -------
            Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries
                of total losses and raw loss values for each objective.

        """
        objectives: Dict[str, float] = {}
        loss_dict: Dict[str, float] = {}

        retain_graph = True

        for i, objective in enumerate(self.losses.keys()):
            # Compute losses
            loss, sub_loss_dict = self.evaluate_objective(objective, context)
            objectives[f"{objective}_objective"] = loss.item()
            loss_dict.update(sub_loss_dict)

            # Skip training if the optimizer is not at a training epoch
            if not self._is_training_epoch(objective, context.epoch):
                continue

            # Set retain_graph to True for all but the last optimizer
            retain_graph = i < len(self.losses) - 1

            # Get parameters for this optimizer
            params = self.params[objective]

            # Compute gradients
            grads = torch.autograd.grad(
                loss, params, create_graph=False, retain_graph=retain_graph
            )

            # Manually update parameters
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad += grad

        # Perform optimization step
        return objectives, loss_dict

    def num_epochs(self) -> int:
        """Get the maximum number of epochs over all optimizers.

        Returns
        -------
            int: The maximum number of epochs across all optimizers.

        """
        return max(self.max_epochs.values())
