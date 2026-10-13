from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from retinal_rl.models.brain import Brain
from retinal_rl.util import rescale_zero_one


def analyze(
    brain: Brain,
    stimuli: dict[str, torch.Tensor],
    circuit_names: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """
    DO NOT USE ATM, not fully implemented yet and may change a lot.
    Run a forward pass of the brain on the given stimuli and record the activity of all circuits.
    Meant to use also internal layers of circuits.
    """
    activity: dict[str, torch.Tensor] = {}
    module_circuit_name_map = {
        str(circuit): name for name, circuit in brain.circuits.items()
    }

    def activity_hook(module: torch.nn.Module, _: torch.Tensor, output: torch.Tensor):
        circuit = module_circuit_name_map.get(str(module), "unknown")
        activity[circuit] = output[0].detach().cpu()

    if circuit_names:
        for name in circuit_names:
            assert name in brain.circuits, f"Circuit {name} not found in brain."
        circuits_to_record = [brain.circuits[name] for name in circuit_names]
    else:
        circuits_to_record = brain.circuits.values()

    # put hooks on all circuits to record activity
    for circuit in circuits_to_record:
        modules_to_hook: list[torch.nn.Module] = []
        for child in circuit.children():
            if isinstance(child, torch.nn.Sequential):
                modules_to_hook.extend(list(child.children()))
            else:
                modules_to_hook.append(child)

        for module in modules_to_hook:
            module.register_forward_hook(activity_hook)

    brain(stimuli)
    for circuit in circuits_to_record:
        circuit._forward_hooks.clear()  # type: ignore
        for module in circuit.children():
            module._forward_hooks.clear()  # type: ignore

    return activity


def fig_setup(
    activity: dict[str, torch.Tensor],
    flatten_activity: bool,
    additional_activity_plots: int,
    num_additional: int,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    mod_3 = num_additional % 3
    two_image_rows = 0 if mod_3 == 0 else 1 if mod_3 == 2 else 2
    two_image_rows = min(two_image_rows, num_additional)
    three_image_rows = max(0, (num_additional - two_image_rows * 2) // 3)
    num_rows = 1 + additional_activity_plots + two_image_rows + three_image_rows

    axes = [[*["raw"] * 2, *["activations_0"] * 4]]

    for i in range(additional_activity_plots):
        axes.append([*["raw"] * 2, *[f"activations_{i+1}"] * 4])

    for i in range(three_image_rows):
        axes.append(
            [
                *[f"additional_{i*3}"] * 2,
                *[f"additional_{i*3+1}"] * 2,
                *[f"additional_{i*3+2}"] * 2,
            ]
        )

    for i in range(two_image_rows):
        axes.append(
            [
                *[f"additional_{(three_image_rows)*3+i*2}"] * 3,
                *[f"additional_{(three_image_rows)*3+i*2+1}"] * 3,
            ]
        )

    if not flatten_activity:
        activity_heights = [act.shape[0] for act in activity.values()]
        activity_heights.extend(
            [np.average(activity_heights)] * (two_image_rows + three_image_rows)
        )
    else:
        activity_heights = None

    fig, ax = plt.subplot_mosaic(
        axes, figsize=(5, num_rows * 2), height_ratios=activity_heights
    )
    return fig, ax


def sort_activity(
    activity_matrix: torch.Tensor, method: str
) -> tuple[torch.Tensor, np.ndarray[np.intp]]:
    # Sort units using tsne
    if method == "tsne":
        tsne = TSNE(n_components=1, random_state=0)
        unit_order = tsne.fit_transform(activity_matrix.T.numpy()).squeeze()
        sorted_indices = np.argsort(unit_order)
        activity_matrix = activity_matrix[:, sorted_indices]
    elif method == "pca":
        pca = PCA(n_components=1, random_state=0)
        unit_order = pca.fit_transform(activity_matrix.T.numpy()).squeeze()
        sorted_indices = np.argsort(unit_order)
        activity_matrix = activity_matrix[:, sorted_indices]
    elif method == "rastermap":
        raise NotImplementedError("Rastermap sorting not implemented yet.")
    else:
        raise ValueError(f"Unknown sorting method: {method}")
    return activity_matrix, sorted_indices


def single_raster_plot(
    activity_matrix: torch.Tensor,
    ylabel: Optional[str] = None,
    cur_frame: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    update_activity: bool = False,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 10))

    # Initialize state attributes if not already present
    if not hasattr(ax, "_plot_im"):
        # Initial setup (only done once)
        img_data = rescale_zero_one(activity_matrix.T.cpu())
        ax._plot_im = ax.imshow(img_data, cmap="gray")

        ax.set_yticks([])
        ax.set_xlabel("Time (frames)")
        ax.set_xlim(0, activity_matrix.size(0))
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if ylabel:
            ax.set_ylabel(ylabel)

        ax._plot_vline = None
    elif update_activity:
        img_data = rescale_zero_one(activity_matrix.T.cpu())
        ax._plot_im.set_data(img_data)

    # Update vertical line
    if cur_frame is not None:
        if ax._plot_vline is not None:
            ax._plot_vline.remove()
        ax._plot_vline = ax.axvline(x=cur_frame, color="red", linestyle="--")

    return ax


def raster_plot(
    activity: dict[str, torch.Tensor],
    flatten_activity: bool = True,
    cur_frame: Optional[int] = None,
    ax: Optional[plt.Axes | List[plt.Axes]] = None,
    update_activity: bool = False,
) -> plt.Axes | List[plt.Axes]:
    if isinstance(ax, plt.Axes):
        ax = [ax]

    activity_keys = list(activity.keys())
    n_plots = len(activity_keys) if not flatten_activity else 1
    assert (
        len(ax) == n_plots or ax is None
    ), f"Number of axes must match number of plots ({n_plots}), but got {len(ax)}."

    if ax is None:
        _, ax = plt.subplots(nrows=n_plots, figsize=(5, n_plots * 2), sharex=True)
        ax = ax.flatten() if n_plots > 1 else [ax]

    # Initialize state on first call
    if not hasattr(ax[0].figure, "_raster_state"):
        ax[0].figure._raster_state = {
            "activity_matrix": None,
            "flatten_activity": flatten_activity,
        }

    state = ax[0].figure._raster_state

    if flatten_activity:
        # Cache the stacked activity matrix
        if state["activity_matrix"] is None or update_activity:
            state["activity_matrix"] = torch.vstack(
                [activity[key] for key in activity_keys]
            )
        activity_matrix = state["activity_matrix"]

        # Single call instead of loop
        single_raster_plot(
            activity_matrix,
            ylabel=", ".join(activity_keys),
            cur_frame=cur_frame,
            ax=ax[0],
        )
    else:
        # Loop only when needed
        for i, (cur_ax, act_key) in enumerate(zip(ax, activity_keys)):
            single_raster_plot(
                activity[act_key],
                ylabel=act_key,
                cur_frame=cur_frame,
                ax=cur_ax,
            )

            if i < n_plots - 1:
                cur_ax.set_xlabel("")  # only show x label on last plot
                cur_ax.sharex(ax[-1])
                plt.setp(cur_ax.get_xticklabels(), visible=False)

    return ax


def full_plot(
    stimuli: dict[str, torch.Tensor],
    activity: dict[str, torch.Tensor],
    flatten_activity: bool = True,
    additional_images: Optional[list[torch.Tensor]] = None,
    additional_titles: Optional[list[str]] = None,
    cur_frame: int = 0,
    num_frames: int = 1000,
    figure: Optional[tuple[plt.Figure, dict]] = None,
) -> tuple[plt.Figure, dict]:
    """
    Plot stimuli and activity data.

    Args:
        stimuli: Dictionary of stimulus tensors
        activity: Dictionary of activity tensors
        flatten_activity: Whether to flatten activity across circuits
        additional_images: List of additional image tensors to display
        additional_titles: List of titles for additional images
        cur_frame: Current frame index
        num_frames: Total number of frames
        figure: Optional tuple of (fig, ax_dict) to update existing figure instead of creating new one

    Returns:
        Tuple of (fig, ax_dict) for potential future updates
    """

    # Determine if we're creating a new figure or updating an existing one
    if figure is None:
        # Create new figure
        additional_activity_plots = 0
        if not flatten_activity:
            num_circuits = len(activity)
            additional_activity_plots = num_circuits - 1

        num_additional = len(additional_images) if additional_images else 0

        fig, ax = fig_setup(
            activity, flatten_activity, additional_activity_plots, num_additional
        )

        # Initialize state
        fig._plot_state = {
            "im_raw": None,
            "vlines": {},
        }
    else:
        # Use existing figure and axes
        fig, ax = figure

    state = fig._plot_state

    # Update vision stimulus
    vision_data = rescale_zero_one(
        stimuli["vision"][0, :num_frames].permute(1, 2, 0).cpu()
    )

    # Update or create the raw stimulus image
    if state["im_raw"] is None:
        state["im_raw"] = ax["raw"].imshow(vision_data)
    else:
        state["im_raw"].set_data(vision_data)
    ax["raw"].axis("off")

    # Update activity plots
    additional_activity_plots = 0
    if not flatten_activity:
        num_circuits = len(activity)
        additional_activity_plots = num_circuits - 1

    raster_plot(
        activity,
        ax=[ax[f"activations_{i}"] for i in range(additional_activity_plots + 1)],
        flatten_activity=flatten_activity,
        cur_frame=cur_frame,
    )

    # Update additional images
    if additional_images:
        for i, img in enumerate(additional_images):
            img_data = rescale_zero_one(img[0].permute(1, 2, 0).cpu().numpy())

            # Update or create image
            if f"im_additional_{i}" not in state:
                state[f"im_additional_{i}"] = ax[f"additional_{i}"].imshow(img_data)
            else:
                state[f"im_additional_{i}"].set_data(img_data)

            ax[f"additional_{i}"].axis("off")

            if additional_titles and len(additional_titles) > i:
                ax[f"additional_{i}"].set_title(additional_titles[i])

        # Hide unused subplots
        if f"additional_{len(additional_images)}" in ax:
            ax[f"additional_{len(additional_images)}"].axis("off")

    fig.tight_layout()
    return fig, ax
