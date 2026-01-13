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
    circuit_name: str = None,
    output_only: bool = True,
) -> dict[str, torch.Tensor]:
    activity: dict[str, torch.Tensor] = {}

    def activity_hook(
        module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        if isinstance(output, tuple):
            output = output[0]  # TODO: Use correct output
        activity[str(module)] = output.detach().cpu()

    if circuit_name:
        assert (
            circuit_name in brain.circuits
        ), f"Circuit {circuit_name} not found in brain."
    circuits_to_record = (
        [brain.circuits[circuit_name]] if circuit_name else brain.circuits.values()
    )
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
    cur_frame: int = 0,
    num_frames: int = 1000,
    return_image: bool = False,
):  # -> Figure:
    fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[1, 2], figsize=(5, 2))
    ax[0].imshow(
        rescale_zero_one(stimuli["vision"][0, :num_frames].permute(1, 2, 0).cpu())
    )
    ax[0].axis("off")

    ax[1].set_title("Activity")
    ax[1].set_xlabel("Time (frames)")
    ax[1].set_ylabel("Units")
    ax[1].set_yticks([])
    ax[1].set_xlim(0, num_frames)
    if isinstance(activity, list):
        activity_matrix = torch.stack(
            [torch.concat([a.flatten() for a in act.values()]) for act in activity
            ]
        )  # (time, units)

        # Sort units using tsne
        tsne = TSNE(n_components=1, random_state=0)
        unit_order = tsne.fit_transform(activity_matrix.T.numpy()).squeeze()
        sorted_indices = np.argsort(unit_order)
        activity_matrix = activity_matrix[:, sorted_indices]

        ax[1].imshow(
            rescale_zero_one(activity_matrix[:num_frames].T.cpu()),
            cmap="gray",
        )
        # draw vertical line at curframe
        ax[1].axvline(x=cur_frame, color="red", linestyle="--")
    else:
        flattened = torch.concat([act.flatten() for act in activity.values()])
        # plot the flattened vector at curframe with colors representing activity
        ax[1].scatter(
            np.repeat(cur_frame, flattened.size(0)),
            np.arange(flattened.size(0)),
            c=flattened.cpu(),
            s=1,
            cmap="gray",
        )

    fig.tight_layout()
    if return_image:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data
    return fig
