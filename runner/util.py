### Imports ###

import logging
import os
import shutil
from typing import Any, Dict, List

import torch
import torch.nn as nn
from omegaconf import DictConfig

from retinal_rl.models.optimizer import BrainOptimizer

# Initialize the logger
log = logging.getLogger(__name__)


def save_checkpoint(
    data_dir: str,
    checkpoint_dir: str,
    max_checkpoints: int,
    brain: nn.Module,
    optimizer: BrainOptimizer,
    histories: dict[str, List[float]],
    completed_epochs: int,
) -> None:
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
