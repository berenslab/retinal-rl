"""Module for managing optimization of complex neural network models with multiple circuits."""

import logging
from typing import Any, Dict, Generic, List, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Optimizer

from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import ContextT, Objective

logger = logging.getLogger(__name__)


class BrainOptimizer(Generic[ContextT]):
    """Manages multiple optimizers that target NeuralCircuits in a Brain.

    This class handles the initialization, state management, and optimization steps
    for multiple optimizers, each associated with specific circuits and objectives.


    Attributes
    ----------
        brain (Brain): The neural network model being optimized.
        optimizers (OrderedDict[str, Optimizer]): Instantiated optimizers, sorted based on connectome.
        objectives (Dict[str, List[Objective]]): Objectives for each optimizer.
        target_circuits (Dict[str, List[str]]): Target circuits for each optimizer.
        min_epochs (Dict[str, int]): Minimum epochs for each optimizer.
        max_epochs (Dict[str, int]): Maximum epochs for each optimizer.

    """

    def __init__(self, brain: Brain, optimizer_configs: Dict[str, DictConfig]):
        """Initialize the BrainOptimizer.

        Args:
        ----
            brain (Brain): The neural network model to optimize.
            optimizer_configs (Dict[str, DictConfig]): Configuration for each optimizer.
                Each config should specify target_circuits, optimizer settings, and objectives.

        Raises:
        ------
            ValueError: If a specified circuit is not found in the brain.

        """
        self.objectives: Dict[str, List[Objective[ContextT]]] = {}
        self.target_circuits: Dict[str, List[str]] = {}
        self.device = next(brain.parameters()).device
        self.min_epochs: Dict[str, int] = {}
        self.max_epochs: Dict[str, int] = {}
        self.optimizers: Dict[str, Optimizer] = {}
        self.params: Dict[str, List[Tensor]] = {}

        # Initialize optimizers in the sorted order
        for name, config in optimizer_configs.items():
            # Collect parameters from target circuits
            params = []
            self.min_epochs[name] = config.get("min_epoch", 0)
            self.max_epochs[name] = config.get("max_epoch", -1)
            if not set(config.target_circuits).issubset(brain.connectome.nodes):
                logger.warning(
                    f"Some target circuits for optimizer {name} are not in the brain's connectome"
                )
            for circuit_name in config.target_circuits:
                if circuit_name in brain.circuits:
                    params.extend(brain.circuits[circuit_name].parameters())

            # Initialize optimizer
            self.optimizers[name] = instantiate(config.optimizer, params=params)
            self.params[name] = params

            # Initialize objectives
            self.objectives[name] = [
                instantiate(obj_config) for obj_config in config.objectives
            ]
            logger.info(
                f"Initalized optimizer: {name}, with objectives: {[obj.key_name for obj in self.objectives[name]]}, and target circuits: {[circuit_name for circuit_name in config.target_circuits]}"
            )

    def compute_loss(
        self, optimizer_name: str, context: ContextT
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss for a specific optimizer.

        Args:
        ----
            optimizer_name (str): Name of the optimizer.
            context (ContextT]): Context information for computing objectives.

        Returns:
        -------
            Tuple[torch.Tensor, Dict[str, float]]: A tuple containing the total loss
                and a dictionary of raw loss values for each objective.

        """
        total_loss = torch.tensor(0.0, device=self.device)
        obj_dict: Dict[str, float] = {}
        for objective in self.objectives[optimizer_name]:
            loss, raw_loss = objective(context)
            total_loss += loss
            obj_dict[objective.key_name] = raw_loss.item()
        return total_loss, obj_dict

    def compute_losses(
        self, context: ContextT
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute losses for all optimizers without performing optimization steps.

        This method is useful for evaluation purposes.

        Args:
        ----
            context (Dict[str, Any]): Context information for computing objectives.

        Returns:
        -------
            Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries
                of total losses and raw loss values for each objective.

        """
        losses: Dict[str, float] = {}
        obj_dict: Dict[str, float] = {}
        for name in self.optimizers.keys():
            loss, sub_obj_dict = self.compute_loss(name, context)
            losses[f"{name}_optimizer_loss"] = loss.item()
            obj_dict.update(sub_obj_dict)
        return losses, obj_dict

    def _is_training_epoch(self, name: str, epoch: int) -> bool:
        """Check if the optimizer should continue training.

        Args:
        ----
            name (str): Name of the optimizer.
            epoch (int): Current epoch number.

        Returns:
        -------
            bool: True if the optimizer should continue training, False otherwise.

        """
        if epoch < self.min_epochs[name]:
            return False
        if self.max_epochs[name] < 0:
            return True
        return epoch < self.max_epochs[name]

    def optimize(self, context: ContextT) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform an optimization step for all optimizers.

        This method computes losses, performs backpropagation, and updates parameters
        for all optimizers.

        Args:
        ----
            context (ContextT): Context information for computing objectives.

        Returns:
        -------
            Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries
                of total losses and raw loss values for each objective.

        """
        losses: Dict[str, float] = {}
        obj_dict: Dict[str, float] = {}

        retain_graph = True

        for i, (name, optimizer) in enumerate(self.optimizers.items()):
            # Compute losses
            loss, sub_obj_dict = self.compute_loss(name, context)
            losses[f"{name}_optimizer_loss"] = loss.item()
            obj_dict.update(sub_obj_dict)

            # Skip training if the optimizer is not at a training epoch
            if not self._is_training_epoch(name, context.epoch):
                continue

            # Get parameters for this optimizer
            params = self.params[name]

            # Set retain_graph to True for all but the last optimizer
            retain_graph = i < len(self.optimizers) - 1

            # Compute gradients
            grads = torch.autograd.grad(
                loss, params, create_graph=False, retain_graph=retain_graph
            )

            # Manually update parameters
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    param.grad = grad

        for optimizer in self.optimizers.values():
            # Perform optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return losses, obj_dict

    def state_dict(self) -> Dict[str, Any]:
        """Get the current state of the BrainOptimizer.

        Returns
        -------
            Dict[str, Any]: A dictionary containing the state of all optimizers
                and their configurations.

        """
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load a state dictionary into the BrainOptimizer.

        This method reinitializes optimizers and objectives based on the loaded state.

        Args:
        ----
            state_dict (Dict[str, Any]): The state dictionary to load.

        """
        # Reinitialize optimizers and objectives
        for name, state_dict in state_dict.items():
            self.optimizers[name].load_state_dict(state_dict)

    def num_epochs(self) -> int:
        """Get the maximum number of epochs over all optimizers.

        Returns
        -------
            int: The maximum number of epochs across all optimizers.

        """
        return max(self.max_epochs.values())
