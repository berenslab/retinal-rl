"""Defines the base class for neural circuits."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, List, Type, get_type_hints

import torch
import torch.nn as nn
import torchscan

from retinal_rl.util import Activation


class NeuralCircuit(nn.Module, ABC):
    """Base class for neural circuits."""

    def __init__(
        self,
        input_shape: List[int],
    ) -> None:
        """Initialize the base model.

        Args:
        ----
        input_shape (List[int]): Shape of the input tensor.

        """
        super().__init__()

        self._input_shape = input_shape

    def __init_subclass__(cls: Type[Any], **kwargs: Any) -> None:
        """Enforces that subclasses have specific parameters in their constructors.

        Args:
        ----
        **kwargs: Additional keyword arguments.

        """
        super().__init_subclass__(**kwargs)

        # Ensure that the __init__ method includes input_shape
        init = cls.__init__
        if not inspect.isfunction(init):
            raise TypeError(f"Class {cls.__name__} does not have a valid __init__ method")

        params = inspect.signature(init).parameters
        if "input_shape" not in params:
            raise TypeError(
                f"Class {cls.__name__} must have 'input_shape' parameters in its __init__ method"
            )

        # Ensure that input_shape has the correct types
        hints = get_type_hints(init)
        if hints.get("input_shape") != List[int]:
            raise TypeError(
                f"Parameter 'input_shape' in class {cls.__name__} must have type 'List[int]'"
            )

    def scan(self) -> None:
        """Run torchscan on the model."""
        torchscan.summary(self, tuple(self.input_shape), receptive_field=True)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural circuit."""
        raise NotImplementedError("Each subclass must implement its own forward method.")

    @property
    def input_shape(self) -> List[int]:
        """Return the shape of the input tensor."""
        return self._input_shape

    @property
    def output_shape(self) -> List[int]:
        """Return the shape of the output tensor."""
        device = next(self.parameters()).device
        with torch.no_grad():
            return list(
                self.forward(torch.zeros(1, *self.input_shape).to(device)).shape[1:]
            )

    @staticmethod
    def str_to_activation(act: str) -> nn.Module:
        """Convert a string to an activation function.

        Args:
        ----
        act (str): The name of the activation function.

        Returns:
        -------
        nn.Module: The activation function.

        """
        act = str.lower(act)
        return Activation[act]()
