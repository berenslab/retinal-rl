import torch

from retinal_rl.models.brain import Brain
from retinal_rl.util import rescale_zero_one
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

import numpy as np


def analyze(
    brain: Brain,
    stimuli: dict[str, torch.Tensor],
    circuit_names: list[str] = None,
    output_only: bool = True,
) -> dict[str, torch.Tensor]:
    activity: dict[str, torch.Tensor] = {}
    module_circuit_name_map = {
        str(circuit): name for name, circuit in brain.circuits.items()
    }

    def activity_hook(
        module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        if isinstance(output, tuple):
            output = output[0]  # TODO: Use correct output
        circuit = module_circuit_name_map.get(str(module), "unknown")
        activity[circuit] = output.detach().cpu()

    if circuit_names:
        for name in circuit_names:
            assert name in brain.circuits, f"Circuit {name} not found in brain."

        circuits_to_record = [brain.circuits[name] for name in circuit_names]
    else:
        circuits_to_record = brain.circuits.values()

    # put hooks on all circuits to record activity
    for circuit in circuits_to_record:
        if output_only:
            circuit.register_forward_hook(activity_hook)
            continue
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


def raster_plot(
    stimuli: dict[str, torch.Tensor],
    activity: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    flatten_activity: bool = True,
    sort_activity: str = "tsne",
    additional_images: list[torch.Tensor] = None, # TODO: Separate out the additional plotting and make this function just for activity raster plots
    additional_titles: list[str] = ["Activity"],
    cur_frame: int = 0,
    num_frames: int = 1000,
    return_image: bool = False,
):  # -> Figure:
    # determine subplot layout based on additional videos
    # first row is always vision and activity, following rows are additional videos if any (2 or 3 columns depending on number of additional videos)

    if isinstance(activity, dict):
        activity = [activity] * num_frames

    additional_activity_plots = 0
    if not flatten_activity:
        num_circuits = len(activity[0])
        additional_activity_plots = num_circuits - 1

    num_additional = len(additional_images) if additional_images else 0
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
        activity_heights = [
            len(activity[0][key].flatten()) for key in activity[0].keys()
        ]
        activity_heights.extend(
            [np.average(activity_heights)] * (two_image_rows + three_image_rows)
        )
    else:
        activity_heights = None

    fig, ax = plt.subplot_mosaic(
        axes, figsize=(5, num_rows * 2), height_ratios=activity_heights
    )
    ax["raw"].imshow(
        rescale_zero_one(stimuli["vision"][0, :num_frames].permute(1, 2, 0).cpu())
    )
    ax["raw"].axis("off")

    # for _ax, title in zip(ax[1:], titles):
    #     _ax.set_title(title)

    # TODO: activity is a dict of circuits. if not in one plot, need to create individual plots.
    # the first dimension is across time in case activity is not available for all frames (yet)
    activity_keys = list(activity[0].keys())
    for i in range(additional_activity_plots + 1):
        cur_ax = ax[f"activations_{i}"]
        if flatten_activity:
            activity_matrix = torch.stack(
                [torch.concat([a.flatten() for a in act.values()]) for act in activity]
            )  # (time, units)
        else:
            # One plot per circuit
            activity_matrix = torch.stack(
                [torch.concat([act[activity_keys[i]].flatten()]) for act in activity]
            )

        # Sort units using tsne
        if sort_activity == "tsne":
            # tsne ordering
            tsne = TSNE(n_components=1, random_state=0)
            unit_order = tsne.fit_transform(activity_matrix.T.numpy()).squeeze()
            sorted_indices = np.argsort(unit_order)
            activity_matrix = activity_matrix[:, sorted_indices]
        elif sort_activity == "pca":
            pca = PCA(n_components=1, random_state=0)
            unit_order = pca.fit_transform(activity_matrix.T.numpy()).squeeze()
            sorted_indices = np.argsort(unit_order)
            activity_matrix = activity_matrix[:, sorted_indices]
        elif sort_activity == "rastermap":
            raise NotImplementedError("Rastermap sorting not implemented yet.")

        cur_ax.imshow(
            rescale_zero_one(activity_matrix[:num_frames].T.cpu()),
            cmap="gray",
        )
        # draw vertical line at curframe
        cur_ax.axvline(x=cur_frame, color="red", linestyle="--")

        cur_ax.set_yticks([])
        cur_ax.set_xlim(0, num_frames)

        if i < additional_activity_plots:
            cur_ax.sharex(ax[f"activations_{additional_activity_plots}"])
        if i == additional_activity_plots:
            cur_ax.set_xlabel("Time (frames)")
        else:
            plt.setp(cur_ax.get_xticklabels(), visible=False)
        cur_ax.spines["top"].set_visible(False)
        cur_ax.spines["bottom"].set_visible(False)
        cur_ax.spines["right"].set_visible(False)
        cur_ax.spines["left"].set_visible(False)
        cur_ax.set_ylabel(
            f"{activity_keys[i]}" if not flatten_activity else ", ".join(activity_keys)
        )

    if additional_images:
        if additional_titles:
            assert len(additional_images) == len(additional_titles), "Number of additional titles must match number of additional images."
        for i, (img, title) in enumerate(zip(additional_images, additional_titles)):
            img = rescale_zero_one(img[0].permute(1, 2, 0).cpu().numpy())
            ax[f"additional_{i}"].imshow(img)
            ax[f"additional_{i}"].set_title(title)
            ax[f"additional_{i}"].axis("off")
        ax[f"additional_{len(additional_images)}"].axis(
            "off"
        )  # if there's only 1 additional image, still hide the last subplot

    fig.tight_layout()
    if return_image:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data
    return fig
