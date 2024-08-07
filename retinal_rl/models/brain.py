from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from retinal_rl.models.neural_circuit import NeuralCircuit


class Brain(nn.Module):
    """The "overarching model" (brain) combining several "partial models" (circuits) - such as encoders, latents, decoders, and task heads - in a specified way."""

    def __init__(
        self,
        name: str,
        circuits: DictConfig,
        sensors: Dict[str, List[int]],
        connections: List[List[str]],
    ) -> None:
        """Brain constructor.

        Args:
        ----
        name: The name of the brain.
        circuits: List of NeuralCircuit objects or dictionaries containing the configurations of the models.
        sensors: Dictionary specifying the names and (tensor) shape of the sensors/stimuli.
        connections: List of pairs of strings specifying the graphical structure of the brain.

        """
        super().__init__()

        self.name = name
        self.circuits : nn.ModuleDict[NeuralCircuit] = nn.ModuleDict()

        self.connectome: nx.DiGraph[str] = nx.DiGraph()
        self.sensors: Dict[str, Tuple[int, ...]] = {}
        for sensor in sensors:
            self.sensors[sensor] = tuple(sensors[sensor])

        for stim in self.sensors:
            self.connectome.add_node(stim)

        for crcnm in circuits:
            self.connectome.add_node(str(crcnm))

        for connection in connections:
            if connection[0] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[0]} not found in the connectome.")
            if connection[1] not in self.connectome.nodes:
                raise ValueError(f"Node {connection[1]} not found in the connectome.")
            self.connectome.add_edge(connection[0], connection[1])

        # Ensure that the connectome is a directed acyclic graph
        if not nx.is_directed_acyclic_graph(self.connectome):
            raise ValueError("The connectome should be a directed acyclic graph.")

        dummy_responses = {
            sensor: torch.rand((1, *self.sensors[sensor])) for sensor in self.sensors
        }

        for node in nx.topological_sort(self.connectome):
            if node not in self.sensors:
                input_tensor = self._calculate_inputs(node, dummy_responses)
                input_shape = list(input_tensor.shape[1:])

                # Check for the output_shape key
                if "output_shape" in circuits[node]:
                    output_shape = circuits[node]["output_shape"]
                    if isinstance(output_shape, str):
                        parts = output_shape.split(".")
                        if len(parts) == 2 and parts[0] in self.circuits:
                            circuit_name, property_name = parts
                            output_shape = getattr(
                                self.circuits[circuit_name], property_name
                            )
                        else:
                            raise ValueError(
                                f"Invalid format or reference in output_shape: {output_shape}"
                            )
                    circuit = instantiate(
                        circuits[node],
                        input_shape=input_shape,
                        output_shape=output_shape,
                    )
                else:
                    circuit = instantiate(circuits[node], input_shape=input_shape)

                dummy_responses[node] = circuit(input_tensor)
                self.circuits[node] = circuit

    def forward(self, stimuli: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        responses: Dict[str, torch.Tensor] = {}

        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                responses[node] = stimuli[node].clone()
            else:
                input = self._calculate_inputs(node, responses)
                responses[node] = self.circuits[node](input)
        return responses

    def scan_circuits(self):
        """Runs torchscan on all circuits and concatenates the reports."""
        # Print connectome
        print("\n\nConnectome:\n")
        print("Nodes: ", self.connectome.nodes)
        print("Edges: ", self.connectome.edges)

        # Run scans on all circuits
        dummy_stimulus: Dict[str, torch.Tensor] = {}
        for sensor in self.sensors:
            dummy_stimulus[sensor] = torch.rand((1, *self.sensors[sensor]), device="cuda")

        for crcnm, circuit in self.circuits.items():
            print(
                f"\n\nCircuit Name: {crcnm}, Class: {circuit.__class__.__name__}, Input Shape: {circuit.input_shape}, Output Shape: {circuit.output_shape}"
            )
            circuit.scan()

    def visualize_connectome(self):
        """Visualize the connectome using networkx and matplotlib."""
        # Initialize a dictionary to hold input and output sizes for visualization
        node_sizes = {}

        # Collect sizes for sensors
        for sensor in self.sensors:
            node_sizes[sensor] = f"Input: {self.sensors[sensor]}"

        # Collect sizes for circuits
        dummy_stimulus: Dict[str, torch.Tensor] = {}
        for sensor in self.sensors:
            dummy_stimulus[sensor] = torch.rand((1, *self.sensors[sensor]))

        for crcnm, circuit in self.circuits.items():
            input_tensor = self._calculate_inputs(crcnm, dummy_stimulus)
            input_shape = list(input_tensor.shape[1:])
            dummy_responses = circuit(input_tensor)
            output_shape = list(dummy_responses.shape[1:])
            node_sizes[crcnm] = f"Input: {input_shape}\nOutput: {output_shape}"

        # Visualize the connectome
        pos = nx.spring_layout(self.connectome)
        plt.figure(figsize=(12, 8))

        # Draw the nodes with their sizes
        nx.draw_networkx_nodes(
            self.connectome, pos, node_color="lightblue", node_size=3000
        )
        nx.draw_networkx_edges(self.connectome, pos, arrows=True)
        nx.draw_networkx_labels(self.connectome, pos, labels=node_sizes, font_size=10)

        plt.title("Connectome Visualization")
        plt.show()

    def _calculate_inputs(
        self, node: str, responses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        inputs: List[torch.Tensor] = []
        input = torch.Tensor()
        for pred in self.connectome.predecessors(node):
            if pred in responses:
                inputs.append(responses[pred])
            else:
                ValueError(f"Input node {pred} to node {node} does not (yet) exist")
        if len(inputs) == 0:
            ValueError(f"No inputs to node {node}")
        elif len(inputs) == 1:
            input = inputs[0]
        else:
            # flatten the inputs and concatenate them
            input = torch.cat([inp.view(inp.size(0), -1) for inp in inputs], dim=1)
        return input
