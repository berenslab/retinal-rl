"""Provides the Brain class, which combines multiple NeuralCircuit instances into a single model by specifying a graph of connections between them."""

import logging
from typing import Any, Dict, List, Tuple, cast

import networkx as nx
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from retinal_rl.models.neural_circuit import NeuralCircuit

logger = logging.getLogger(__name__)

from retinal_rl.models.neural_circuit import NeuralCircuit


class Brain(nn.Module):
    """The "overarching model" (brain) combining several "partial models" (circuits) - such as encoders, latents, decoders, and task heads specified by a graph."""

    def __init__(
        self,
        circuits: Dict[str, DictConfig],
        sensors: Dict[str, List[int]],
        connections: List[List[str]],
    ) -> None:
        """Initialize the brain with a set of circuits, sensors, and connections.

        Args:
        ----
        circuits: A dictionary of circuit configurations.
        sensors: A dictionary of sensor names and their dimensions.
        connections: A list of connections between sensors and circuits.

        """
        super().__init__()

        # Initialize attributes
        self.circuits: Dict[str, NeuralCircuit] = {}
        self._module_dict: nn.ModuleDict = nn.ModuleDict()
        self.connectome: nx.DiGraph[str] = nx.DiGraph()
        self.sensors: Dict[str, Tuple[int, ...]] = {}
        for sensor in sensors:
            self.sensors[sensor] = tuple(sensors[sensor])

        # Build the connectome
        self.connectome.add_nodes_from(self.sensors.keys())
        self.connectome.add_nodes_from(circuits.keys())
        for connection in connections:
            if connection[0] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[0]} not found in the connectome.")
            if connection[1] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[1]} not found in the connectome.")
            self.connectome.add_edge(connection[0], connection[1])

        # Ensure that the connectome is a directed acyclic graph
        if not nx.is_directed_acyclic_graph(self.connectome):
            raise ValueError("The connectome should be a directed acyclic graph.")

        # Create dummy responses to help calculate the input shape for each neural circuit
        dummy_responses = {
            sensor: torch.rand((1, *self.sensors[sensor])) for sensor in self.sensors
        }

        # Instantiate the neural circuits
        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                continue

            input_tensor = self._assemble_inputs(node, dummy_responses)
            input_shape = list(input_tensor.shape[1:])
            circuit_config = OmegaConf.to_container(circuits[node], resolve=True)
            circuit_config = cast(Dict[str, Any], circuit_config)

            # Check for an explicit output_shape key
            if "output_shape" in circuit_config:
                output_shape = circuit_config["output_shape"]
                if isinstance(output_shape, str):
                    output_shape = self._resolve_output_shape(output_shape)
                circuit = instantiate(
                    circuit_config,
                    input_shape=input_shape,
                    output_shape=output_shape,
                )
            else:
                circuit = instantiate(
                    circuit_config,
                    input_shape=input_shape,
                )

            # Set attribute to register the module with pytorch
            dummy_responses[node] = circuit(input_tensor)

            self.circuits[node] = circuit
            self._module_dict[node] = circuit

    def _resolve_output_shape(self, output_shape: str) -> Tuple[int, ...]:
        parts = output_shape.split(".")
        if len(parts) == 2 and parts[0] in self.circuits:
            circuit_name, property_name = parts
            return getattr(self.circuits[circuit_name], property_name)
        raise ValueError(
            f"Invalid format for output_shape: {output_shape}. Must be of the form 'circuit_name.property_name'"
        )

    def forward(self, stimuli: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass of the brain. Computed by following the connectome from sensors through the circuits."""
        responses: Dict[str, Tensor] = {}

        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                responses[node] = stimuli[node]
            else:
                input = self._assemble_inputs(node, responses)
                responses[node] = self.circuits[node](input)
        return responses

    def scan_circuits(self):
        """Run torchscan on all circuits and concatenates the reports."""
        # Print connectome
        print("\n\nConnectome:\n")
        print("Nodes: ", self.connectome.nodes)
        print("Edges: ", self.connectome.edges)

        # Run scans on all circuits
        dummy_stimulus: Dict[str, Tensor] = {}
        for sensor in self.sensors:
            dummy_stimulus[sensor] = torch.rand((1, *self.sensors[sensor]))

        for crcnm, circuit in self.circuits.items():
            print(
                f"\n\nCircuit Name: {crcnm}, Class: {circuit.__class__.__name__}, Input Shape: {circuit.input_shape}, Output Shape: {circuit.output_shape}"
            )
            circuit.scan()

    def _assemble_inputs(self, node: str, responses: Dict[str, Tensor]) -> Tensor:
        """Assemble the inputs to a given node by concatenating the responses of its predecessors."""
        inputs: List[Tensor] = []
        input = Tensor()
        for pred in self.connectome.predecessors(node):
            if pred in responses:
                inputs.append(responses[pred])
            else:
                raise ValueError(f"Input node {pred} to node {node} does not (yet) exist")
        if len(inputs) == 0:
            raise ValueError(f"No inputs to node {node}")
        if len(inputs) == 1:
            input = inputs[0]
        else:
            # flatten the inputs and concatenate them
            input = torch.cat([inp.view(inp.size(0), -1) for inp in inputs], dim=1)
        return input
