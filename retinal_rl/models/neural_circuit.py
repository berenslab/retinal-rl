"""Defines the base class for neural circuits."""

from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

from retinal_rl.util import Activation


class NeuralCircuit(nn.Module, ABC):
    """Base class for neural circuits."""

    def __init__(
        self,
        input_shapes: list[list[int]],
    ) -> None:
        """Initialize the base neural circuit.

        Design Decision: Hybrid Tuple/List Approach
        ------------------------------------------
        - Shape specifications: list[list[int]] for readability and Hydra compatibility
        - Runtime tensors: tuple[Tensor, ...] for Python conventions and external tool compatibility

        This gives us:
        - Clean, readable type hints: list[list[int]] vs tuple[tuple[int, ...], ...]
        - Natural Hydra integration without custom resolvers
        - Standard Python tuple returns for multiple outputs
        - Compatibility with external tools (captum, torchinfo)

        Args:
        ----
        input_shapes: List of input tensor shapes for this circuit.
                     Each inner list represents the shape of one input tensor.
                     Example: [[3, 32, 32]] for single RGB image input
                             [[3, 32, 32], [128]] for image + state inputs

        """
        super().__init__()

        # Convert to tuple format for internal consistency
        self._input_shapes = tuple(tuple(shape) for shape in input_shapes)

    @abstractmethod
    @beartype
    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Forward pass of the neural circuit.

        Args:
        ----
        inputs: Tuple of input tensors

        Returns:
        -------
        Tuple of output tensors
        """
        raise NotImplementedError(
            "Each subclass must implement its own forward method."
        )

    @property
    def input_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Return the shapes of the input tensors."""
        return self._input_shapes

    @property
    def output_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Return the shapes of the output tensors."""
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy_inputs = tuple(
                torch.zeros(1, *shape).to(device) for shape in self.input_shapes
            )
            outputs = self.forward(dummy_inputs)
            return tuple(tuple(output.shape[1:]) for output in outputs)

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


class SimpleNeuralCircuit(NeuralCircuit):
    """Base class for neural circuits with single primary input and output."""

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the primary input shape for convenience."""
        return self.input_shapes[0]

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the primary output shape for convenience."""
        return self.output_shapes[0]
