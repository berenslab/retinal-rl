import torch
import numpy as np

from retinal_rl.models.brain import Brain
from retinal_rl.util import rescale_zero_one
from sklearn.decomposition import PCA


def analyze(
    brain: Brain,
    stimuli: dict[str, torch.Tensor],
    num_pcs: int = 3,
    rescale_per_frame: bool = False,
    pca: PCA | None = None,
) -> dict[str, torch.Tensor]:
    assert num_pcs > 0, "num_pcs must be positive"

    outputs = brain(stimuli)
    reduced_outputs: dict[str, torch.Tensor] = {}
    for key in outputs:
        if key not in ["retina", "thalamus", "v1"]:
            continue
        # Reduce number of outputs via PCA along channel dimension, keep other dimensions
        output = outputs[key][0].detach().cpu() # TODO: Use correct item in tuple
        num_channels = output.size(1)
        actual_num_pcs = min(num_pcs, num_channels)
        reduced_outputs[key] = torch.empty(output.size(0), actual_num_pcs, *output.size()[2:])

        pca = None
        for frame_no, frame in enumerate(output):
            reduced = single_frame_pca(frame, num_pcs=actual_num_pcs, pca=pca)
            reduced_tensor = torch.tensor(reduced)
            if rescale_per_frame:
                for i in range(reduced_tensor.shape[0]):
                    reduced_tensor[i] = rescale_zero_one(reduced_tensor[i])
            reduced_outputs[key][frame_no] = reduced_tensor

    return reduced_outputs

def single_frame_pca(frame: torch.Tensor, num_pcs: int = 3, pca: PCA | None = None) -> np.ndarray:
    flattened = frame.view(frame.size(0), -1)  # (channels, height*width)

    #Reduce number of channels to num_pcs via PCA
    if pca is None:
        pca = PCA(n_components=num_pcs)
        reduced = pca.fit_transform(flattened.T).T  # (num_pcs, height*width)
    else:
        reduced = pca.transform(flattened.T).T  # (num_pcs, height*width)
    return reduced.reshape(num_pcs, *frame.size()[1:])  # (num_pcs, height, width)


def plot():  # -> Figure:
    # TODO: Implement plotting logic
    raise NotImplementedError
