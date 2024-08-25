import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from retinal_rl.models.neural_circuit import NeuralCircuit

logger = logging.getLogger(__name__)


class Brain(nn.Module):
    """The "overarching model" (brain) combining several "partial models" (circuits) - such as encoders, latents, decoders, and task heads - in a specified way."""

    def __init__(
        self,
        name: str,
        circuits: Dict[str, DictConfig],
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

        self.connectome: nx.DiGraph[str] = nx.DiGraph()
        self.sensors: Dict[str, Tuple[int, ...]] = {}
        for sensor in sensors:
            self.sensors[sensor] = tuple(sensors[sensor])

        # Build the connectome
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

        # Create dummy responses to help calculate the input shape for each neural circuit
        dummy_responses = {
            sensor: torch.rand((1, *self.sensors[sensor])) for sensor in self.sensors
        }

        # Instantiate the neural circuits
        for node in nx.topological_sort(self.connectome):
            if node not in self.sensors:
                input_tensor = self._assemble_inputs(node, dummy_responses)
                input_shape = list(input_tensor.shape[1:])

                circuit_config = OmegaConf.to_container(circuits[node], resolve=True)
                # circuit_config = circuits[node]
                optimizer_config = circuit_config.pop("optimizer", None)

                # Check for an explicit output_shape key
                if "output_shape" in circuit_config:
                    output_shape = circuit_config["output_shape"]
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
                        circuit_config,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        _recursive_=False,
                    )
                else:
                    circuit = instantiate(
                        circuit_config,
                        input_shape=input_shape,
                        _recursive_=False,
                    )

                # Set attribute to register the module with pytorch
                setattr(self, f"_circuit_{name}", circuit)
                dummy_responses[node] = circuit(input_tensor)

                if optimizer_config:
                    optimizer = instantiate(optimizer_config, circuit.parameters())
                    circuit.optimizer = optimizer
                self.circuits[node] = circuit

    def forward(self, stimuli: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        responses: Dict[str, torch.Tensor] = {}

        for node in nx.topological_sort(self.connectome):
            if node in self.sensors:
                responses[node] = stimuli[node].clone()
            else:
                input = self._assemble_inputs(node, responses)
                responses[node] = self.circuits[node](input)
        return responses

    def optimize(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        final_losses: Dict[str, float] = {}
        for circuit_name, circuit in self.circuits.items():
            if circuit.optimizer:
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
                circuit.optimizer.zero_grad()
                total_loss.backward()
                circuit.optimizer.step()

                # Store the final loss
                final_losses[circuit_name] = total_loss.item()

            else:
                # For circuits without optimizers, just clean up any gradients
                for param in circuit.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        return final_losses

    def get_optimizer_states(self) -> Dict[str, Any]:
        optimizer_states: Dict[str, Any] = {}
        for name, circuit in self.circuits.items():
            state = circuit.get_optimizer_state()
            if state:
                optimizer_states[name] = state
        return optimizer_states

    def load_optimizer_states(self, optimizer_states: Dict[str, Any]) -> None:
        for name, state in optimizer_states.items():
            if name in self.circuits:
                self.circuits[name].load_optimizer_state(state)
            else:
                logger.warning(
                    f"Circuit '{name}' not found in the current Brain instance."
                )

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

    def _assemble_inputs(
        self, node: str, responses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Assemble the inputs to a given node by concatenating the responses of its predecessors."""
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
