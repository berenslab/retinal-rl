"""Provides the Brain class, which combines multiple NeuralCircuit instances into a single model by specifying a graph of connections between them."""

import logging
from io import StringIO
from typing import Dict, List, OrderedDict, Tuple

import networkx as nx
import torch
from networkx.classes.digraph import DiGraph
from torch import Tensor, nn
from torchinfo import summary

from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder
from retinal_rl.models.neural_circuit import NeuralCircuit

logger = logging.getLogger(__name__)


class Brain(nn.Module):
    """The "overarching model" (brain) combining several "partial models" (circuits) - such as encoders, latents, decoders, and task heads specified by a graph."""

    def __init__(
        self,
        circuits: Dict[str, NeuralCircuit],
        sensors: Dict[str, List[int]],
        connectome: DiGraph,  # type: ignore
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
        self.circuits: Dict[str, NeuralCircuit] = circuits
        self._module_dict: nn.ModuleDict = nn.ModuleDict(circuits)
        self.connectome: DiGraph[str] = connectome
        self.sensors: Dict[str, Tuple[int, ...]] = {}
        for sensor in sensors:
            self.sensors[sensor] = tuple(sensors[sensor])

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

    def scan(self) -> str:
        """
        Performs a comprehensive scan of the model and its circuits, returning the results as a string.

        Returns:
            str: A formatted string containing the complete scan results
        """
        output = StringIO()
        device = next(self.parameters()).device

        # Create dummy stimulus
        dummy_stimulus: Dict[str, Tensor] = {
            sensor: torch.rand((1, *self.sensors[sensor]), device=device)
            for sensor in self.sensors
        }

        # Whole brain scan
        output.write("\nWhole Brain Scan:\n\n")
        model_stats = summary(self, input_data=[dummy_stimulus], verbose=0)
        output.write(str(model_stats))

        # Circuit scans
        output.write("\n\n\nCircuit Scans:\n")

        for circuit_name, circuit in self.circuits.items():
            output.write(
                f"\n\nCircuit Name: {circuit_name}"
                f"\nClass: {circuit.__class__.__name__}"
                f"\nInput Shape: {circuit.input_shape}"
                f"\nOutput Shape: {circuit.output_shape}\n"
            )

            circuit_stats = summary(
                circuit, (1, *tuple(circuit.input_shape)), verbose=0
            )
            output.write(str(circuit_stats))

        # Get the complete output as a string
        result = output.getvalue()
        output.close()

        return result

    def _assemble_inputs(self, node: str, responses: Dict[str, Tensor]) -> Tensor:
        """Assemble the inputs to a given node by concatenating the responses of its predecessors."""
        inputs: List[Tensor] = []
        input = Tensor()
        for pred in self.connectome.predecessors(node):
            if pred in responses:
                inputs.append(responses[pred])
            else:
                raise ValueError(
                    f"Input node {pred} to node {node} does not (yet) exist"
                )
        if len(inputs) == 0:
            raise ValueError(f"No inputs to node {node}")
        if len(inputs) == 1:
            input = inputs[0]
        else:
            # flatten the inputs and concatenate them
            input = torch.cat([inp.view(inp.size(0), -1) for inp in inputs], dim=1)
        return input


def get_cnn_circuit(
    brain: Brain,
) -> Tuple[Tuple[int, ...], OrderedDict[str, nn.Module]]:
    """Find the longest path starting from a sensor, along a path of ConvolutionalEncoders. This likely won't work very well for particularly complex graphs."""
    cnn_paths: List[List[str]] = []

    # Create for the subgraph of sensors and cnns
    cnn_dict: Dict[str, ConvolutionalEncoder] = {}
    for node, circuit in brain.circuits.items():
        if isinstance(circuit, ConvolutionalEncoder):
            cnn_dict[node] = circuit

    cnn_nodes = list(cnn_dict.keys())
    sensor_nodes = [node for node in brain.sensors]
    subgraph: nx.DiGraph[str] = nx.DiGraph(
        nx.subgraph(brain.connectome, cnn_nodes + sensor_nodes)
    )
    end_nodes: List[str] = [
        node for node in cnn_nodes if not list(subgraph.successors(node))
    ]

    for sensor in sensor_nodes:
        for end_node in end_nodes:
            cnn_paths.extend(
                nx.all_simple_paths(subgraph, source=sensor, target=end_node)
            )

    # find the longest path
    path = max(cnn_paths, key=len)
    logger.info(f"Convolutional circuit path for analysis: {path}")
    # Split off the sensor node
    sensor, *path = path
    # collect list of cnns
    cnn_circuits: List[ConvolutionalEncoder] = [cnn_dict[node] for node in path]
    # Combine all cnn layers
    tuples: List[Tuple[str, nn.Module]] = []
    for circuit in cnn_circuits:
        for name, module in circuit.conv_head.named_children():
            tuples.extend([(name, module)])

    input_shape = brain.sensors[sensor]
    return input_shape, OrderedDict(tuples)
