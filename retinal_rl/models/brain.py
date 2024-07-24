from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer

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
        self.circuits: Dict[str, NeuralCircuit] = {}
        self.optimizers: Dict[str, Optimizer] = {}

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
                optimizer_config = circuits[node].pop("optimizer", None)

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
                setattr(self, f"_circuit_{name}", circuit)
                if optimizer_config is not None:
                    self.optimizers[node] = instantiate(
                        optimizer_config, params=self.circuits[node].parameters()
                    )

    def forward(self, stimuli: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        responses: Dict[str, torch.Tensor] = {}

        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                responses[node] = stimuli[node].clone()
            else:
                input = self._calculate_inputs(node, responses)
                responses[node] = self.circuits[node](input)
        return responses

    def optimize(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        final_losses: Dict[str, float] = {}
        for circuit_name, circuit in self.circuits.items():
            if circuit_name in self.optimizers:
                # Compute total loss for this circuit
                total_loss = torch.tensor(
                    0.0, device=next(circuit.parameters()).device, requires_grad=True
                )
                for loss_type, loss_value in loss_dict.items():
                    if loss_type in circuit.loss_weights:
                        total_loss += circuit.loss_weights[loss_type] * loss_value

                # Add L1 weight regularization
                if "l1_weight" in circuit.reg_weights:
                    l1_weight = circuit.reg_weights["l1_weight"]
                    l1_reg = torch.mean(
                        torch.stack([p.abs().mean() for p in circuit.parameters()])
                    )
                    total_loss += l1_weight * l1_reg

                if "l2_weight" in circuit.reg_weights:
                    l2_weight = circuit.reg_weights["l2_weight"]
                    l2_reg = torch.mean(
                        torch.stack([p.pow(2).mean() for p in circuit.parameters()])
                    )
                    total_loss += l2_weight * torch.sqrt(l2_reg)

                # Backward pass and optimize
                self.optimizers[circuit_name].zero_grad()
                total_loss.backward()
                self.optimizers[circuit_name].step()

                # Store the final loss
                final_losses[circuit_name] = total_loss.item()

            else:
                # For circuits without optimizers, just clean up any gradients
                for param in circuit.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        return final_losses

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        for name, optimizer in self.optimizers.items():
            state_dict[prefix + f"optimizer_{name}"] = optimizer.state_dict()

        return destination

    def load_state_dict(
        self,
        state_dict: Union[Dict[str, Any], OrderedDict[str, torch.Tensor]],
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        """Load the state dict of the Brain, including optimizers."""
        optimizer_state_dict = {}
        model_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if key.startswith("optimizer_"):
                optimizer_name = key[10:]  # Remove "optimizer_" prefix
                optimizer_state_dict[optimizer_name] = value
            else:
                model_state_dict[key] = value

        # Load the model state
        incompatible_keys = super().load_state_dict(model_state_dict, strict=strict)

        # Load optimizer states
        for name, optimizer in self.optimizers.items():
            if name in optimizer_state_dict:
                optimizer.load_state_dict(optimizer_state_dict[name])
            elif strict:
                raise RuntimeError(f"Optimizer '{name}' not found in the checkpoint")

        return incompatible_keys

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

    # def visualize_connectome(self):
    #     """Visualize the connectome using networkx and matplotlib."""
    #     # Initialize a dictionary to hold input and output sizes for visualization
    #     node_sizes = {}
    #
    #     # Collect sizes for sensors
    #     for sensor in self.sensors:
    #         node_sizes[sensor] = f"Input: {self.sensors[sensor]}"
    #
    #     # Collect sizes for circuits
    #     dummy_stimulus: Dict[str, torch.Tensor] = {}
    #     for sensor in self.sensors:
    #         dummy_stimulus[sensor] = torch.rand((1, *self.sensors[sensor]))
    #
    #     for crcnm, circuit in self.circuits.items():
    #         input_tensor = self._calculate_inputs(crcnm, dummy_stimulus)
    #         input_shape = list(input_tensor.shape[1:])
    #         dummy_responses = circuit(input_tensor)
    #         output_shape = list(dummy_responses.shape[1:])
    #         node_sizes[crcnm] = f"Input: {input_shape}\nOutput: {output_shape}"
    #
    #     # Visualize the connectome
    #     pos = nx.spring_layout(self.connectome)
    #     plt.figure(figsize=(12, 8))
    #
    #     # Draw the nodes with their sizes
    #     nx.draw_networkx_nodes(
    #         self.connectome, pos, node_color="lightblue", node_size=3000
    #     )
    #     nx.draw_networkx_edges(self.connectome, pos, arrows=True)
    #     nx.draw_networkx_labels(self.connectome, pos, labels=node_sizes, font_size=10)
    #
    #     plt.title("Connectome Visualization")
    #     plt.show()
