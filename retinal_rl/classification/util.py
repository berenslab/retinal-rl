### Imports ###

import os
import shutil
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from retinal_rl.models.brain import Brain


def initialize(
    data_path: str,
    checkpoint_path: str,
    plot_path: str,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    completed_epochs = 0
    if os.path.exists(data_path):
        print("Data path exists. Loading existing model and history.")
        brain, optimizer, history, completed_epochs = load_checkpoint(
            data_path, brain, optimizer
        )
    else:
        print("Data path does not exist. Initializing new model and history.")
        history = initialize_histories()
        # create the directories
        os.makedirs(data_path)
        os.makedirs(checkpoint_path)
        os.makedirs(plot_path)

    return brain, optimizer, history, completed_epochs


def save_checkpoint(
    data_path: str,
    checkpoint_path: str,
    max_checkpoints: int,
    brain: nn.Module,
    optimizer: Optimizer,
    history: dict[str, List[float]],
    completed_epochs: int,
) -> None:
    current_file = os.path.join(data_path, "current_checkpoint.pt")
    checkpoint_file = os.path.join(
        checkpoint_path, f"checkpoint_epoch_{completed_epochs}.pt"
    )
    checkpoint_dict = {
        "completed_epochs": completed_epochs,
        "model_state_dict": brain.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_history": history,
    }

    # Save checkpoint
    torch.save(checkpoint_dict, checkpoint_file)

    # Copy the checkpoint to current_brain.pt
    shutil.copyfile(checkpoint_file, current_file)

    # Remove older checkpoints if the number exceeds the threshold
    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint_epoch_")],
        key=lambda x: int(x.split("_")[2].split(".")[0]),
        reverse=True,
    )
    while len(checkpoints) > max_checkpoints:
        os.remove(os.path.join(checkpoint_path, checkpoints.pop()))


def load_checkpoint(
    data_path: str,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    # Load histories from the JSON file
    checkpoint_file = os.path.join(data_path, "current_checkpoint.pt")

    # check if files don't exist
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"File not found: {checkpoint_file}")

    # Load the state dict into the brain model
    checkpoint = torch.load(checkpoint_file)
    brain.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epochs = checkpoint["completed_epochs"]
    history = checkpoint["training_history"]

    return brain, optimizer, history, completed_epochs


def delete_results(experiment_path: str, data_path: str) -> None:
    """Delete the data directory after prompting the user for confirmation.

    Args:
    ----
        experiment_dir (str): Path to the experiment directory.
        data_path (str): Path to the data directory to be deleted.

    """
    full_path = os.path.join(experiment_path, data_path)

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
        "train_fraction_correct": [],
        "train_reconstruction": [],
        "test_total": [],
        "test_classification": [],
        "test_fraction_correct": [],
        "test_reconstruction": [],
    }
