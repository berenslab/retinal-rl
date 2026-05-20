import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.figure import Figure

from retinal_rl.models.brain import Brain
from retinal_rl.plot.util import plot_histories
from retinal_rl.rl.analysis.plot import rescale_range
from retinal_rl.util import NumpyEncoder


class FigureLogger:
    def __init__(
        self, use_wandb: bool, plot_dir: Path, checkpoint_plot_dir: Path, run_dir: Path
    ):
        self.use_wandb = use_wandb
        self.plot_dir = plot_dir
        self.checkpoint_plot_dir = checkpoint_plot_dir
        self.run_dir = run_dir

    def log_figure(
        self,
        fig: Figure,
        sub_dir: str,
        file_name: str,
        epoch: int,
        copy_checkpoint: bool,
    ) -> None:
        if self.use_wandb:
            title = f"{self._wandb_title(sub_dir)}/{self._wandb_title(file_name)}"
            img = wandb.Image(fig)
            wandb.log({title: img}, commit=False)
        else:
            self.save_figure(sub_dir, file_name, fig)
            if copy_checkpoint:
                self._checkpoint_copy(sub_dir, file_name, epoch)
        plt.close(fig)

    @staticmethod
    def _wandb_title(title: str) -> str:
        # Split the title by slashes
        parts = title.split("/")

        def capitalize_part(part: str) -> str:
            # Split the part by dashes
            words = part.split("_")
            # Capitalize each word
            capitalized_words = [word.capitalize() for word in words]
            # Join the words with spaces
            return " ".join(capitalized_words)

        # Capitalize each part, then join with slashes
        capitalized_parts = [capitalize_part(part) for part in parts]
        return "/".join(capitalized_parts)

    def _checkpoint_copy(self, sub_dir: str, file_name: str, epoch: int) -> None:
        src_path = self.plot_dir / sub_dir / f"{file_name}.png"

        dest_dir = self.checkpoint_plot_dir / f"epoch_{epoch}" / sub_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{file_name}.png"

        shutil.copy2(src_path, dest_path)

    def save_figure(self, sub_dir: str, file_name: str, fig: Figure) -> None:
        dir = self.plot_dir / sub_dir
        dir.mkdir(exist_ok=True)
        file_path = dir / f"{file_name}.png"
        fig.savefig(file_path)

    def plot_and_save_histories(
        self, histories: dict[str, list[float]], save_always: bool = False
    ):
        if not self.use_wandb or save_always:
            hist_fig = plot_histories(histories)
            self.save_figure("", "histories", hist_fig)
            plt.close(hist_fig)

    def save_summary(self, brain: Brain):
        summary = brain.scan()
        filepath = self.run_dir / "brain_summary.txt"
        filepath.write_text(summary)

        if self.use_wandb:
            wandb.save(str(filepath), base_path=self.run_dir, policy="now")

    def save_dict(
        self, path: Path, store_dict: dict[str, Any], compressed: bool = True
    ):
        if compressed:
            compressed_dict: dict[str, Any] = {}
            for key, value in store_dict.items():
                if isinstance(value, np.ndarray):
                    compressed_dict[key] = rescale_range(
                        value, center_zero=True, out_max=255
                    ).astype(np.uint8)
                elif isinstance(value, list) and all(
                    isinstance(item, np.ndarray) for item in value
                ):
                    compressed_dict[key] = [
                        rescale_range(item, center_zero=True, out_max=255).astype(
                            np.uint8
                        )
                        for item in value
                    ]
                else:
                    compressed_dict[key] = value
            np.savez_compressed(path, **compressed_dict)
        else:
            with open(path, "w") as f:
                json.dump(store_dict, f, cls=NumpyEncoder)
