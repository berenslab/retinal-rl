"""t-SNE visualization of VAE bottleneck (LGN) layer activations."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from retinal_rl.analysis.plot import FigureLogger
from retinal_rl.classification.imageset import Imageset
from retinal_rl.models.brain import Brain


def analyze(
    device: torch.device,
    brain: Brain,
    test_set: Imageset,
    layer_name: str,
    max_samples: int = 1000,
    batch_size: int = 64,
    perplexity: int = 30,
    n_iter: int = 300,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Analyze bottleneck layer with t-SNE.

    Args:
        device: torch device
        brain: Brain model
        test_set: Imageset to visualize
        layer_name: Name of the layer to visualize (e.g. "visual_cortex")
        max_samples: Maximum samples to use
        batch_size: Batch size for dataloader
        perplexity: t-SNE perplexity
        n_iter: Number of t-SNE iterations

    Returns:
        Tuple of (tsne_results, labels)
    """
    brain.eval()
    brain.to(device)

    # Check circuit exists
    if layer_name not in brain.circuits:
        print(
            f"Warning: Circuit '{layer_name}' not found. "
            f"Available circuits: {list(brain.circuits.keys())}"
        )
        return None, None

    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    activations_list = []
    labels_list = []
    sample_count = 0

    print(f"Collecting bottleneck activations (max {max_samples})...")
    with torch.no_grad():
        for src, img, label in dataloader:
            if sample_count >= max_samples:
                break

            img = img.to(device)
            stimulus = {"vision": img}
            responses = brain(stimulus)

            # Extract the bottleneck activation
            bottleneck_output = responses[layer_name][0]

            # Flatten if needed
            batch_size = bottleneck_output.shape[0]
            flat_activation = bottleneck_output.view(batch_size, -1)

            activations_list.append(flat_activation.cpu().numpy())
            labels_list.extend(label.cpu().tolist())

            sample_count += batch_size

    activations = np.vstack(activations_list)
    labels = np.array(labels_list[: len(activations)])

    print(f"Computing t-SNE on {activations.shape[0]} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, activations.shape[0] // 3),
        max_iter=n_iter,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    tsne_results = tsne.fit_transform(activations)

    return tsne_results, labels


def plot(
    log: FigureLogger,
    tsne_results: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    copy_checkpoint: bool,
    layer_name: str = "visual_cortex",
) -> Figure:
    """
    Create and log t-SNE visualization.

    Args:
        log: FigureLogger instance
        tsne_results: t-SNE coordinates (n_samples, 2)
        labels: Class labels (n_samples,)
        epoch: Current epoch
        copy_checkpoint: Whether to copy to checkpoint dir
        layer_name: Name of the visualized layer

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=[color],
            label=f"Class {label}",
            alpha=0.7,
            s=50,
        )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title(f"{layer_name} t-SNE")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    log.log_figure(
        fig,
        "latent_visualization",
        f"{layer_name}_tsne",
        epoch,
        copy_checkpoint,
    )

    return fig
