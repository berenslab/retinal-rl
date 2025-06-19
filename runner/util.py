"""Utility functions for the runner module."""

### Imports ###

import logging
import os
import shutil
from pathlib import Path
from typing import Any, cast

import networkx as nx
import torch
from hydra.utils import instantiate
from networkx.classes import DiGraph
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.optimizer import Optimizer

from retinal_rl.models.brain import Brain, assemble_inputs
from retinal_rl.models.neural_circuit import NeuralCircuit

nx.DiGraph.__class_getitem__ = classmethod(lambda _, __: "nx.DiGraph")  # type: ignore

# Initialize the logger
log = logging.getLogger(__name__)


def save_checkpoint(
    data_dir: Path,
    checkpoint_dir: Path,
    max_checkpoints: int,
    brain: nn.Module,
    optimizer: Optimizer,
    histories: dict[str, list[float]],
    completed_epochs: int,
) -> None:
    """Save a checkpoint of the model and optimizer state."""
    current_file = data_dir / "current_checkpoint.pt"
    checkpoint_file = checkpoint_dir / f"epoch_{completed_epochs}.pt"
    checkpoint_dict: dict[str, Any] = {
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


def load_brain_weights(brain: Brain, checkpoint_path: str):
    actual_state_dict = brain.state_dict()
    ckpt = torch.load(checkpoint_path)
    if "brain_state_dict" in ckpt:  # classification / our logging file
        checkpoint = ["brain_state_dict"]
        for key in checkpoint:
            if key in actual_state_dict and "fc" not in key:
                actual_state_dict[key] = checkpoint[key]
    elif "model" in ckpt:  # Sample Factory file
        checkpoint = ckpt["model"]

        for key in checkpoint:
            if "brain" in key:
                actual_state_dict[key[6:]] = checkpoint[key]
    else:
        raise ValueError(
            "Checkpoint does not contain 'brain_state_dict' or 'brain' key."
        )

    brain.load_state_dict(actual_state_dict)


def delete_results(run_dir: Path) -> None:
    """Delete the results directory."""

    if not run_dir.exists():
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


def create_brain(brain_cfg: DictConfig) -> Brain:
    sensors = OmegaConf.to_container(brain_cfg.sensors, resolve=True)
    sensors = cast(dict[str, list[int]], sensors)

    connections = OmegaConf.to_container(brain_cfg.connections, resolve=True)
    connections = cast(list[list[str]], connections)

    connectome, circuits = assemble_neural_circuits(
        brain_cfg.circuits, sensors, connections
    )

    return Brain(circuits, sensors, connectome)


def import_class(import_path):  # TODO: Move to more general utils
    parts = import_path.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]

    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _create_dummy_responses(
    sensor_shapes: dict[str, tuple[int, ...]],
) -> dict[str, tuple[torch.Tensor, ...]]:
    # Create dummy responses to help calculate the input shape for each neural circuit
    # Now returns tuples to match our new Brain.forward() interface
    dummy_responses = {
        sensor: (torch.rand((1, *sensor_shapes[sensor])),) for sensor in sensor_shapes
    }
    if "rnn_state" in sensor_shapes:
        shape = (1, *sensor_shapes["rnn_state"])
        dummy_responses["rnn_state"] = (torch.rand(shape),)
    return dummy_responses


def assemble_neural_circuits(
    circuits: DictConfig,
    sensors: dict[str, list[int]],
    connections: list[list[str]],
) -> tuple[DiGraph[str], dict[str, NeuralCircuit]]:
    """
    Assemble a dictionary of neural circuits based on the provided configurations.
    """
    assembled_circuits: dict[str, NeuralCircuit] = {}
    connectome: DiGraph[str] = nx.DiGraph()
    sensor_shapes: dict[str, tuple[int, ...]] = {
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

    dummy_responses = _create_dummy_responses(sensor_shapes)

    # Instantiate the neural circuits
    for node in nx.topological_sort(connectome):
        if node in sensor_shapes:
            continue

        circuit_config = OmegaConf.select(circuits, node)

        _circuit_class = import_class(circuit_config._target_)
        inputs = assemble_inputs(node, connectome, dummy_responses)

        # Convert input shapes to new tuple format expected by circuits
        # TODO: automatic retrieval of all input shapes?
        input_shapes = tuple(tuple(int(dim) for dim in inp.shape[1:]) for inp in inputs)
        

        # Check for an explicit output_shape key
        if "output_shape" in circuit_config:
            output_shape = circuit_config["output_shape"]
            if isinstance(output_shape, str):
                output_shape = _resolve_output_shape(output_shape, assembled_circuits)
            circuit = instantiate(
                circuit_config,
                input_shapes=input_shapes,
                output_shape=output_shape,
                _convert_="partial",
            )
        else:
            circuit = instantiate(
                circuit_config,
                input_shapes=input_shapes,
                _convert_="partial",
            )

        # Update dummy responses with tuple interface
        output_tuple = circuit(inputs)
        dummy_responses[node] = output_tuple
        # TODO: Review: Order of inputs might not be correct, at least there are no guarantees
        # perhaps implicitly defined through the connectome config (also not a nice way though)

        assembled_circuits[node] = circuit

    return connectome, assembled_circuits


def _resolve_output_shape(
    output_shape: str, circuits: dict[str, "NeuralCircuit"]
) -> tuple[int, ...]:
    """Resolve the output shape from a string reference."""
    parts = output_shape.split(".")
    if len(parts) == 2 and parts[0] in circuits:
        circuit_name, property_name = parts
        return getattr(circuits[circuit_name], property_name)
    raise ValueError(
        f"Invalid format for output_shape: {output_shape}. Must be of the form 'circuit_name.property_name'"
    )


def search_conf(config: DictConfig | dict, search_str: str) -> list:
    """
    Recursively search for strings in a DictConfig.

    Args:
        config (omegaconf.DictConfig): The configuration to search.

    Returns:
        list: A list of all values containing the string.
    """
    found_values = []

    def traverse_config(cfg):
        for key, value in cfg.items():
            if isinstance(value, (dict, DictConfig)):
                traverse_config(value)
            elif isinstance(value, str) and search_str in value:
                found_values.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and search_str in item:
                        found_values.append(item)

    traverse_config(config)
    return found_values
