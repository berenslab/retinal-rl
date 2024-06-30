### Imports ###

import json
import os
import shutil
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from retinal_rl.models.brain import Brain


def initialize(
    data_path: str, checkpoint_path: str, plot_path: str, brain: Brain
) -> Tuple[Brain, Dict[str, List[float]]]:
    """Load or initializing the model and histories.

    Args:
    ----
        data_path (str): Path to the data directory.
        checkpoint_path (str): Path to the checkpoint directory.
        plot_path (str): Path to the plot directory.
        brain (Brain): The initialized Brain.

    """
    if os.path.exists(data_path):
        print("Data path exists. Loading existing model and history.")
        brain, history = load_results(data_path, brain)
    else:
        print("Data path does not exist. Initializing new model and history.")
        history = initialize_histories()
        # create the directories
        os.makedirs(data_path)
        os.makedirs(checkpoint_path)
        os.makedirs(plot_path)

    return brain, history


def save_results(
    data_path: str,
    checkpoint_path: str,
    max_checkpoints: int,
    brain: nn.Module,
    history: dict[str, List[float]],
) -> None:
    """Save the model and training history to files.

    Args:
    ----
        data_path (str): Path to the data directory.
        checkpoint_path (str): Path to the checkpoint directory.
        max_checkpoints (int): Maximum number of checkpoints to keep.
        brain (nn.Module): The trained model.
        history (dict[str, List[float]]): The training history.

    """
    # Save histories as a single JSON file
    histories_file_path = os.path.join(data_path, "histories.json")
    with open(histories_file_path, "w") as f:
        json.dump(history, f)

    total_epochs = len(history["train_total"])
    brain_file_path = os.path.join(data_path, "current_brain.pt")
    brain_checkpoint_path = os.path.join(
        checkpoint_path, f"checkpoint_epoch_{total_epochs}.pt"
    )

    # Save checkpoint
    torch.save(brain.state_dict(), brain_checkpoint_path)

    # Copy the checkpoint to current_brain.pt
    shutil.copyfile(brain_checkpoint_path, brain_file_path)

    # Remove older checkpoints if the number exceeds the threshold
    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint_epoch_")],
        key=lambda x: int(x.split("_")[2].split(".")[0]),
        reverse=True,
    )
    while len(checkpoints) > max_checkpoints:
        os.remove(os.path.join(checkpoint_path, checkpoints.pop()))


def load_results(
    data_path: str,
    brain: Brain,
) -> Tuple[Brain, Dict[str, List[float]]]:
    """Load the model and training history from the saved files.

    Args:
    ----
        data_path (str): Path to the data directory.
        brain (Brain): The initialized Brain.

    Returns:
    -------
        Tuple[nn.Module, Dict[str, List[float]]]: The loaded brain model and training history.

    """
    # Load histories from the JSON file
    histories_file_path = os.path.join(data_path, "histories.json")
    brain_file_path = os.path.join(data_path, "current_brain.pt")

    # check if files don't exist
    if not os.path.exists(histories_file_path):
        raise FileNotFoundError(f"File not found: {histories_file_path}")
    if not os.path.exists(brain_file_path):
        raise FileNotFoundError(f"File not found: {brain_file_path}")

    with open(histories_file_path, "r") as f:
        history = json.load(f)

    # Load the state dict into the brain model
    brain.load_state_dict(torch.load(brain_file_path))

    return brain, history


def delete_results(experiment_dir: str, data_path: str) -> None:
    """Delete the data directory after prompting the user for confirmation.

    Args:
    ----
        experiment_dir (str): Path to the experiment directory.
        data_path (str): Path to the data directory to be deleted.

    """
    full_path = os.path.join(experiment_dir, data_path)

    if not os.path.exists(data_path):
        print(f"Directory {data_path} does not exist.")
        return

    confirmation = input(
        f"Are you sure you want to delete the directory {full_path}? (Y/N): "
    )

    if confirmation.lower() == "y":
        try:
            shutil.rmtree(data_path)
            print(f"Directory {data_path} has been deleted.")
        except Exception as e:
            print(f"An error occurred while deleting the directory: {e}")
    else:
        print("Deletion cancelled.")


def initialize_histories() -> Dict[str, List[float]]:
    """Initialize an empty training history."""
    return {
        "train_total": [],
        "train_classification": [],
        "train_reconstruction": [],
        "test_total": [],
        "test_classification": [],
        "test_reconstruction": [],
    }
