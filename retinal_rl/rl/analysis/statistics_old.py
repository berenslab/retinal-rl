import numpy as np
import torch
from captum.attr import NeuronGradient
from openTSNE import TSNE
from tqdm import tqdm

from retinal_rl.util import encoder_out_size, rf_size_and_start


def gaussian_noise_stas(cfg, env, actor_critic, nbtch, nreps, prgrs):
    """
    Returns the receptive fields of every layer of a convnet as computed by spike-triggered averaging.
    """

    enc = actor_critic.encoder.vision_model
    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    nclrs, hght, wdth = list(env.observation_space["obs"].shape)
    ochns = nclrs

    btchsz = [nbtch, nclrs, hght, wdth]

    stas = {}

    repttl = len(enc.conv_head) * nreps
    mdls = []

    with torch.no_grad(), tqdm(
        total=repttl, desc="Generating STAs", disable=not (prgrs)
    ) as pbar:
        for lyrnm, mdl in enc.conv_head.named_children():
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

            for _ in range(nreps):
                pbar.update(1)

                for j in range(ochns):
                    obss = torch.randn(size=btchsz, device=dev)
                    obss1 = obss[:, :, hmn:hmx, wmn:wmx].cpu()
                    outs = subenc(obss)[:, j, hidx, widx].cpu()

                    if torch.sum(outs) != 0:
                        stas[lyrnm][j] += (
                            np.average(obss1, axis=0, weights=outs) / nreps
                        )

    return stas


def gradient_receptive_fields(cfg, env, actor_critic, prgrs):
    """
    Returns the receptive fields of every layer of a convnet as computed by neural gradients.
    """

    enc = actor_critic.encoder.vision_model
    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    nclrs, hght, wdth = list(env.observation_space["obs"].shape)
    ochns = nclrs

    imgsz = [1, nclrs, hght, wdth]
    # Obs requires grad
    obs = torch.zeros(size=imgsz, device=dev, requires_grad=True)

    stas = {}

    repttl = len(enc.conv_head)
    mdls = []

    with torch.no_grad(), tqdm(
        total=repttl, desc="Generating Attributions", disable=not (prgrs)
    ) as pbar:
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

            pbar.update(1)

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


def row_zscore(mat):
    return (mat - np.mean(mat, 1)[:, np.newaxis]) / (
        np.std(mat, 1)[:, np.newaxis] + 1e-8
    )


def fit_tsne_1d(data):
    print("fitting 1d-tSNE...")
    # default openTSNE params
    tsne = TSNE(
        n_components=1,
        perplexity=20,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    return tsne.fit(data)


def fit_tsne(data):
    print("fitting tSNE...")
    # default openTSNE params
    tsne = TSNE(
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    return tsne.fit(data.T)


def get_stim_coll(all_health, health_dep=-8, death_dep=30):
    stim_coll = np.diff(all_health)
    stim_coll[stim_coll == health_dep] = 0  # excluding 'hunger' decrease
    stim_coll[stim_coll > death_dep] = 0  # excluding decrease due to death
    stim_coll[stim_coll < -death_dep] = 0
    return stim_coll
