"""Utility functions for the runner module."""

### Imports ###

import logging
import os
import shutil
from typing import Any, Dict, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
from hydra.utils import instantiate
from networkx.classes import DiGraph
from omegaconf import DictConfig, OmegaConf
from torch.optim.optimizer import Optimizer

from retinal_rl.models.neural_circuit import NeuralCircuit

nx.DiGraph.__class_getitem__ = classmethod(lambda a, b: "nx.DiGraph")  # type: ignore

# Initialize the logger
log = logging.getLogger(__name__)


def save_checkpoint(
    data_dir: str,
    checkpoint_dir: str,
    max_checkpoints: int,
    brain: nn.Module,
    optimizer: Optimizer,
    histories: dict[str, List[float]],
    completed_epochs: int,
) -> None:
    """Save a checkpoint of the model and optimizer state."""
    current_file = os.path.join(data_dir, "current_checkpoint.pt")
    checkpoint_file = os.path.join(checkpoint_dir, f"epoch_{completed_epochs}.pt")
    checkpoint_dict: Dict[str, Any] = {
        "completed_epochs": completed_epochs,
        "brain_state_dict": brain.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "histories": histories,
    }

    # Save checkpoint
    torch.save(checkpoint_dict, checkpoint_file)

    # Copy the checkpoint to current_brain.pt
    shutil.copyfile(checkpoint_file, current_file)

    # Remove older checkpoints if the number exceeds the threshold
    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
        reverse=True,
    )
    while len(checkpoints) > max_checkpoints:
        os.remove(os.path.join(checkpoint_dir, checkpoints.pop()))


def delete_results(cfg: DictConfig) -> None:
    """Delete the results directory."""
    run_dir: str = cfg.system.run_dir

    if not os.path.exists(run_dir):
        print(f"Directory {run_dir} does not exist.")
        return

    confirmation = input(
        f"Are you sure you want to delete the directory {run_dir}? (Y/N): "
    )

    if confirmation.lower() == "y":
        try:
            shutil.rmtree(run_dir)
            print(f"Directory {run_dir} has been deleted.")
        except Exception as e:
            print(f"An error occurred while deleting the directory: {e}")
    else:
        print("Deletion cancelled.")


def assemble_neural_circuits(
    circuits: DictConfig,
    sensors: Dict[str, List[int]],
    connections: List[List[str]],
) -> Tuple[DiGraph[str], Dict[str, NeuralCircuit]]:
    """
    Assemble a dictionary of neural circuits based on the provided configurations.
    """
    assembled_circuits: Dict[str, "NeuralCircuit"] = {}
    connectome: DiGraph[str] = nx.DiGraph()
    sensor_shapes: Dict[str, Tuple[int, ...]] = {
        sensor: tuple(sensors[sensor]) for sensor in sensors
    }
    # get unique names in connections without sensors
    circuit_names = set([connection[1] for connection in connections])

    # Build the connectome
    connectome.add_nodes_from(sensor_shapes.keys())
    connectome.add_nodes_from(circuit_names)
    for connection in connections:
        if connection[0] not in connectome.nodes:
            raise ValueError(f"Node {connection[0]} not found in the connectome.")
        if connection[1] not in connectome.nodes:
            raise ValueError(f"Node {connection[1]} not found in the connectome.")
        connectome.add_edge(connection[0], connection[1])

    # Ensure that the connectome is a directed acyclic graph
    if not nx.is_directed_acyclic_graph(connectome):
        raise ValueError("The connectome should be a directed acyclic graph.")

    # Create dummy responses to help calculate the input shape for each neural circuit
    dummy_responses = {
        sensor: torch.rand((1, *sensor_shapes[sensor])) for sensor in sensor_shapes
    }

    # Instantiate the neural circuits
    for node in nx.topological_sort(connectome):
        if node in sensor_shapes:
            continue

        print(f"Processing node {node}")
        circuit_config = OmegaConf.select(circuits, node)
        input_tensor = _assemble_inputs(node, connectome, dummy_responses)
        input_shape = list(input_tensor.shape[1:])

        # Check for an explicit output_shape key
        if "output_shape" in circuit_config:
            output_shape = circuit_config["output_shape"]
            if isinstance(output_shape, str):
                output_shape = _resolve_output_shape(output_shape, assembled_circuits)
            circuit = instantiate(
                circuit_config,
                input_shape=input_shape,
                output_shape=output_shape,
                _convert_="partial",
            )
        else:
            circuit = instantiate(
                circuit_config,
                input_shape=input_shape,
                _convert_="partial",
            )

        # Update dummy responses
        dummy_responses[node] = circuit(input_tensor)

        assembled_circuits[node] = circuit

    return connectome, assembled_circuits


def _assemble_inputs(
    node: str,
    connectome: DiGraph[str],
    responses: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Assemble the inputs to a given node by concatenating the responses of its predecessors."""
    inputs: List[torch.Tensor] = []
    for pred in connectome.predecessors(node):
        if pred in responses:
            inputs.append(responses[pred])
        else:
            raise ValueError(f"Input node {pred} to node {node} does not (yet) exist")
    if len(inputs) == 0:
        raise ValueError(f"No inputs to node {node}")
    if len(inputs) == 1:
        return inputs[0]
    # flatten the inputs and concatenate them
    return torch.cat([inp.view(inp.size(0), -1) for inp in inputs], dim=1)


def _resolve_output_shape(
    output_shape: str, circuits: Dict[str, "NeuralCircuit"]
) -> Tuple[int, ...]:
    """Resolve the output shape from a string reference."""
    parts = output_shape.split(".")
    if len(parts) == 2 and parts[0] in circuits:
        circuit_name, property_name = parts
        return getattr(circuits[circuit_name], property_name)
    raise ValueError(
        f"Invalid format for output_shape: {output_shape}. Must be of the form 'circuit_name.property_name'"
    )
