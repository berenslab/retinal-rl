from captum.attr import NeuronGradient
import numpy as np
import torch

from retinal_rl.models.brain import Brain
from retinal_rl.util import encoder_out_size, rf_size_and_start


def gradient_receptive_fields(device: torch.device, brain: Brain):
    """
    Returns the receptive fields of every layer of a convnet as computed by neural gradients.
    """

    nclrs, hght, wdth = brain.stimuli["vision"]
    ochns = nclrs

    imgsz = [1, nclrs, hght, wdth]
    # Obs requires grad
    obs = torch.zeros(size=imgsz, device=device, requires_grad=True)

    stas = {}

    repttl = len(brain.conv_head)
    mdls = []

    with torch.no_grad():

        for lyrnm, mdl in enc.conv_head.named_children():

            gradient_calculator = NeuronGradient(enc, mdl)
            mdls.append(mdl)
            subenc = torch.nn.Sequential(*mdls)

            # check if mdl has out channels
            if hasattr(mdl, "out_channels"):
                ochns = mdl.out_channels
            hsz, wsz = encoder_out_size(subenc, hght, wdth)

            hidx = (hsz - 1) // 2
            widx = (wsz - 1) // 2

            hrf_size, wrf_size, hmn, wmn = rf_size_and_start(subenc, hidx, widx)

            hmx = hmn + hrf_size
            wmx = wmn + wrf_size

            stas[lyrnm] = np.zeros((ochns, nclrs, hrf_size, wrf_size))

            for j in range(ochns):
                grad = (
                    gradient_calculator.attribute(obs, (j, hidx, widx))[
                        0, :, hmn:hmx, wmn:wmx
                    ]
                    .cpu()
                    .numpy()
                )

                stas[lyrnm][j] = grad

    return stas
