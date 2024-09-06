"""Module for managing optimization of complex neural network models with multiple circuits."""

from typing import Any, Dict, List, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer

from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import Objective


class BrainOptimizer:
    """Manages multiple optimizers that target NeuralCircuits in a Brain.

    This class handles the initialization, state management, and optimization steps
    for multiple optimizers, each associated with specific circuits and objectives.

    Attributes
    ----------
        brain (Brain): The neural network model being optimized.
        optimizers (Dict[str, Optimizer]): Instantiated optimizers.
        objectives (Dict[str, List[Objective]]): Objectives for each optimizer.
        target_circuits (Dict[str, List[str]]): Target circuits for each optimizer.

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
        self.optimizers: Dict[str, Optimizer] = {}
        self.objectives: Dict[str, List[Objective]] = {}
        self.target_circuits: Dict[str, List[str]] = {}
        self.device = next(brain.parameters()).device

        for name, config in optimizer_configs.items():
            # Collect parameters from target circuits
            params = []
            self.target_circuits[name] = config.target_circuits
            for circuit_name in config.target_circuits:
                if circuit_name in brain.circuits:
                    params.extend(brain.circuits[circuit_name].parameters())
                else:
                    raise ValueError(
                        f"Circuit: {circuit_name} associated with the Optimizer: {name} not found."
                    )

            # Initialize optimizer
            self.optimizers[name] = instantiate(config.optimizer, params=params)

            # Initialize objectives
            self.objectives[name] = [
                instantiate(obj_config) for obj_config in config.objectives
            ]

    def compute_loss(
        self, optimizer_name: str, context: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss for a specific optimizer.

        Args:
        ----
            optimizer_name (str): Name of the optimizer.
            context (Dict[str, torch.Tensor]): Context information for computing objectives.

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
        self, context: Dict[str, Any]
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

    def optimize(
        self, context: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Perform an optimization step for all optimizers.

        This method computes losses, performs backpropagation, and updates parameters
        for all optimizers.

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
        retain_graph = True
        for i, (name, optimizer) in enumerate(self.optimizers.items()):
            if i == len(self.optimizers) - 1:
                retain_graph = False
            optimizer.zero_grad()
            loss, sub_obj_dict = self.compute_loss(name, context)
            loss.backward(retain_graph=retain_graph)
            losses[f"{name}_optimizer_loss"] = loss.item()
            obj_dict.update(sub_obj_dict)

        for name, optimizer in self.optimizers.items():
            optimizer.step()

        return losses, obj_dict

    def state_dict(self) -> Dict[str, Any]:
        """Get the current state of the BrainOptimizer.

        Returns
        -------
            Dict[str, Any]: A dictionary containing the state of all optimizers
                and their configurations.

        """
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state_dicts: Dict[str, Any]) -> None:
        """Load a state dictionary into the BrainOptimizer.

        This method reinitializes optimizers and objectives based on the loaded state.

        Args:
        ----
            state_dict (Dict[str, Any]): The state dictionary to load.

        """
        # Reinitialize optimizers and objectives
        for name, state_dict in state_dicts.items():
            self.optimizers[name].load_state_dict(state_dict)
