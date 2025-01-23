"""Utility functions for plotting the results of statistical analyses."""

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import wandb
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import MaxNLocator

from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import ContextT, Objective
from retinal_rl.util import FloatArray, NumpyEncoder


def make_image_grid(arrays: list[FloatArray], nrow: int) -> FloatArray:
    """Create a grid of images from a list of numpy arrays."""
    # Assuming arrays are [C, H, W]
    n = len(arrays)
    if not n:
        return np.array([])

    ncol = nrow
    nrow = (n - 1) // ncol + 1

    nchns, hght, wdth = arrays[0].shape
    grid = np.zeros((nchns, hght * nrow, wdth * ncol))

    for idx, array in enumerate(arrays):
        i = idx // ncol
        j = idx % ncol
        grid[:, i * hght : (i + 1) * hght, j * wdth : (j + 1) * wdth] = array

    return grid


def plot_brain_and_optimizers(brain: Brain, objective: Objective[ContextT]) -> Figure:
    graph = brain.connectome

    # Compute the depth of each node
    depths: dict[str, int] = {}
    for node in nx.topological_sort(graph):
        depths[node] = (
            max([depths[pred] for pred in graph.predecessors(node)] + [-1]) + 1
        )

    # Create a position dictionary based on depth
    pos: dict[str, tuple[float, float]] = {}
    nodes_at_depth: dict[int, list[str]] = {}
    for node, depth in depths.items():
        if depth not in nodes_at_depth:
            nodes_at_depth[depth] = []
        nodes_at_depth[depth].append(node)

    max_depth = max(depths.values())
    for depth, nodes in nodes_at_depth.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (
                (i - width / 2) / (width + 1),
                -(max_depth - depth) / max_depth,
            )

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color="gray", arrows=True, ax=ax)

    # Color scheme for different node types
    color_map = {"sensor": "lightblue", "circuit": "lightgreen"}

    # Generate colors for losses
    loss_colors = sns.color_palette("husl", len(objective.losses))

    # Draw nodes
    for node in graph.nodes():
        x, y = pos[node]

        # Determine node type and base color
        if node in brain.sensors:
            base_color = color_map["sensor"]
            targeting_losses = []
        else:
            base_color = color_map["circuit"]
            targeting_losses = [
                loss
                for loss in objective.losses
                if (node in loss.target_circuits or (loss.target_circuits == "__all__"))
            ]

        # Draw base circle
        circle = Circle((x, y), 0.05, facecolor=base_color, edgecolor="black")
        ax.add_patch(circle)

        # Determine which losses target this node

        if targeting_losses:
            # Calculate angle for each loss
            angle_per_loss = 360 / len(targeting_losses)

            # Draw a wedge for each targeting loss
            for i, loss in enumerate(targeting_losses):
                start_angle = i * angle_per_loss
                wedge = Wedge(
                    (x, y),
                    0.07,
                    start_angle,
                    start_angle + angle_per_loss,
                    width=0.02,
                    facecolor=loss_colors[objective.losses.index(loss)],
                )
                ax.add_patch(wedge)

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold", ax=ax)

    # Add a legend for losses
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Loss: {loss.__class__.__name__}",
            markerfacecolor=color,
            markersize=15,
        )
        for loss, color in zip(objective.losses, loss_colors)
    ]

    # Add legend elements for sensor and circuit
    legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Sensor",
                markerfacecolor=color_map["sensor"],
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Circuit",
                markerfacecolor=color_map["circuit"],
                markersize=15,
            ),
        ]
    )

    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title("Brain Connectome and Loss Targets")
    plt.tight_layout()
    plt.axis("equal")
    plt.axis("off")

    return fig


def plot_receptive_field_sizes(
    input_shape: tuple[int, ...], rf_layers: dict[str, FloatArray]
) -> Figure:
    """Plot the receptive field sizes for each layer of the convolutional part of the network."""
    # Get visual field size from the input shape
    [_, height, width] = list(input_shape)

    # Calculate receptive field sizes for each layer
    rf_sizes: list[tuple[int, int]] = []
    layer_names: list[str] = []
    for name, rf in rf_layers.items():
        rf_height, rf_width = rf.shape[2:]
        rf_sizes.append((rf_height, rf_width))
        layer_names.append(name)

    rf_sizes.reverse()
    layer_names.reverse()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Reverse y-axis to match image coordinates
    ax.set_aspect("equal", "box")
    ax.set_title("Receptive Field Sizes of Convolutional Layers")

    # Set up grid lines
    ax.set_xticks(np.arange(0, width + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, height + 1, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    # Set up major ticks and labels
    major_ticks_x = [0, width // 2, width]
    major_ticks_y = [0, height // 2, height]
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticklabels(["0", f"{width // 2}", f"{width}"])
    ax.set_yticklabels([f"{height}", f"{height // 2}", "0"])

    # Use a color palette from seaborn
    colors = sns.color_palette("husl", n_colors=len(rf_sizes))

    # Plot receptive fields
    for (rf_height, rf_width), color, name in zip(rf_sizes, colors, layer_names):
        center_x, center_y = width // 2, height // 2
        rect = patches.Rectangle(
            (center_x - rf_width // 2, center_y - rf_height // 2),
            rf_width,
            rf_height,
            fill=True,
            facecolor=color,
            edgecolor=color,
            label=f"{name} ({rf_height}x{rf_width})",
        )
        ax.add_patch(rect)

    # Add legend
    ax.legend()

    plt.tight_layout()

    return fig


def plot_histories(histories: dict[str, list[float]]) -> Figure:
    """Plot training and test losses over epochs."""
    train_metrics = [
        key.split("_", 1)[1] for key in histories if key.startswith("train_")
    ]
    test_metrics = [
        key.split("_", 1)[1] for key in histories if key.startswith("test_")
    ]

    # Use the intersection of train and test metrics to ensure we have both for each metric
    metrics = list(set(train_metrics) & set(test_metrics))

    # Determine the number of rows needed (2 metrics per row)
    num_rows = (len(metrics) + 1) // 2

    fig: Figure
    axs: list[Axes]
    fig, axs = plt.subplots(
        num_rows, 2, figsize=(15, 5 * num_rows), constrained_layout=True
    )
    axs = axs.flatten() if num_rows > 1 else [axs]

    for idx, metric in enumerate(metrics):
        ax: Axes = axs[idx]
        lbl: str = " ".join([word.capitalize() for word in metric.split("_")])

        if f"train_{metric}" in histories:
            ax.plot(
                histories[f"train_{metric}"],
                label=f"{lbl} Training",
                color="black",
            )
        if f"test_{metric}" in histories:
            ax.plot(
                histories[f"test_{metric}"],
                label=f"{lbl} Test",
                color="red",
            )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.set_title(lbl)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Remove any unused subplots
    for idx in range(len(metrics), len(axs)):
        fig.delaxes(axs[idx])

    fig.suptitle("Training and Test Metrics", fontsize=16)
    return fig


def set_integer_ticks(ax: Axes):
    """Set integer ticks for both x and y axes."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


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

    def save_dict(self, path: Path, dict: dict[str, Any]):
        with open(path, "w") as f:
            json.dump(dict, f, cls=NumpyEncoder)
