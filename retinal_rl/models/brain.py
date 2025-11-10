"""Provides the Brain class, which combines multiple NeuralCircuit instances into a single model by specifying a graph of connections between them."""

import logging
from collections import OrderedDict
from io import StringIO

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
        circuits: dict[str, NeuralCircuit],
        sensors: dict[str, list[int]],
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
        self.circuits: dict[str, NeuralCircuit] = circuits
        self._module_dict: nn.ModuleDict = nn.ModuleDict(circuits)
        self.connectome: DiGraph[str] = connectome
        self.sensors: dict[str, tuple[int, ...]] = {}
        for sensor in sensors:
            self.sensors[sensor] = tuple(sensors[sensor])

    def forward(self, stimuli: dict[str, Tensor]) -> dict[str, tuple[Tensor, ...]]:
        """Forward pass of the brain. Computed by following the connectome from sensors through the circuits."""
        responses: dict[str, tuple[Tensor, ...]] = {}

        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                responses[node] = (stimuli[node],)  # Wrap sensor inputs as tuples
            else:
                input_tuple = assemble_inputs(node, self.connectome, responses)
                output_tuple = self.circuits[node](input_tuple)
                responses[node] = output_tuple  # Store full output tuple
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
        dummy_stimulus: dict[str, Tensor] = {
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
                f"\nInput Shapes: {circuit.input_shapes}"
                f"\nOutput Shapes: {circuit.output_shapes}\n"
            )

            # Create dummy input tuple for torchinfo
            dummy_inputs = tuple(
                torch.zeros(1, *shape, device=device) for shape in circuit.input_shapes
            )
            circuit_stats = summary(circuit, input_data=[dummy_inputs], verbose=0)
            output.write(str(circuit_stats))

        # Get the complete output as a string
        result = output.getvalue()
        output.close()

        return result

    def get_circuit_by_type(self, circuit_type: type[nn.Module]) -> list[str]:
        """Get all circuits of a specific type."""
        return [
            name
            for name, circuit in self.circuits.items()
            if isinstance(circuit, circuit_type)
        ]


def get_cnn_circuit(
    brain: Brain,
) -> tuple[tuple[int, ...], OrderedDict[str, nn.Module]]:
    """Find the longest path starting from a sensor, along a path of ConvolutionalEncoders. This likely won't work very well for particularly complex graphs."""
    cnn_paths: list[list[str]] = []

    # Create for the subgraph of sensors and cnns
    cnn_dict: dict[str, ConvolutionalEncoder] = {}
    for node, circuit in brain.circuits.items():
        if isinstance(circuit, ConvolutionalEncoder):
            cnn_dict[node] = circuit

    cnn_nodes = list(cnn_dict.keys())
    sensor_nodes = [node for node in brain.sensors]
    subgraph: nx.DiGraph[str] = nx.DiGraph(
        nx.subgraph(brain.connectome, cnn_nodes + sensor_nodes)
    )
    end_nodes: list[str] = [
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
    cnn_circuits: list[ConvolutionalEncoder] = [cnn_dict[node] for node in path]
    # Combine all cnn layers
    tuples: list[tuple[str, nn.Module]] = []
    for circuit in cnn_circuits:
        for name, module in circuit.conv_head.named_children():
            tuples.extend([(name, module)])

    input_shape = brain.sensors[sensor]
    return input_shape, OrderedDict(tuples)


def assemble_inputs(
    node: str,
    connectome: DiGraph,  # type: ignore
    responses: dict[str, tuple[Tensor, ...]],
) -> tuple[Tensor, ...]:
    """Assemble the inputs to a given node from responses of its predecessors."""
    inputs: list[Tensor] = []
    for pred in connectome.predecessors(node):
        if pred in responses:
            # Take the first (primary) output from each predecessor
            inputs.append(responses[pred][0])
        else:
            raise ValueError(f"Input node {pred} to node {node} does not (yet) exist")
    if len(inputs) == 0:
        raise ValueError(f"No inputs to node {node}")

    return tuple(inputs)
