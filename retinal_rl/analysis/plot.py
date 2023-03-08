import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
plt.style.use('misc/default.mplstyle')

from matplotlib.animation import FuncAnimation
from retinal_rl.analysis.statistics import fit_tsne_1d,get_stim_coll,row_zscore
from retinal_rl.analysis.util import normalize_data


from tqdm.auto import tqdm

greyscale = np.array([0.299, 0.587, 0.114])

def simulation_plot(sim_recs,frame_step=0,animate=False,fps=35):

    imgs0 = sim_recs["imgs"]
    nimgs0 = sim_recs["nimgs"]
    attrs0 = sim_recs["attrs"]
    hlths0 = sim_recs["hlths"]
    crwds0 = sim_recs["crwds"]
    dns = sim_recs["dns"]
    dndices = np.where(dns)

    imgs = normalize_data(imgs0)
    gimgs0 = np.average(imgs,axis=2,weights=greyscale)

    gimgs = np.array([gimgs0,gimgs0,gimgs0]).transpose(1,2,0,3)
    nimgs = normalize_data(nimgs0)
    attrs1 = normalize_data(attrs0)
    attrs = gimgs + attrs1

    crwds = np.ma.array(crwds0,mask=dns)
    hlths = np.ma.array(hlths0,mask=dns)

    if not animate:
        img0 = imgs[:, :, :, frame_step]
        nimg0 = nimgs[:, :, :, frame_step]
        attr0 = attrs[:, :, :, frame_step]
    else:
        img0 = imgs[:, :, :, 0]
        nimg0 = nimgs[:, :, :, 0]
        attr0 = attrs[:, :, :, 0]

    t_max = imgs.shape[3]

    mosaic = """
    aabb
    aacd
    ddee
    ddee
    """

    fig, ax_dict = plt.subplot_mosaic(
        mosaic,
        figsize=(6, 3),
    )

    imax = ax_dict["a"]
    rwdax = ax_dict["b"]
    hlthax = ax_dict["c"]
    nimax = ax_dict["e"]
    attax = ax_dict["d"]

    trng = np.linspace(0, t_max - 1, t_max)

    # FoV
    imax.set_title("Field of View")
    imax.set_xticks([])
    imax.set_yticks([])
    im = imax.imshow(img0,interpolation=None)
    imax.spines["top"].set_visible(True)
    imax.spines["right"].set_visible(True)

    # Normalized FoV
    nimax.set_title("Normalized FoV")
    nimax.set_xticks([])
    nimax.set_yticks([])
    nim = nimax.imshow(nimg0,interpolation=None)
    nimax.spines["top"].set_visible(True)
    nimax.spines["right"].set_visible(True)

    # Attribution
    attax.set_title("Attribution")
    attax.set_xticks([])
    attax.set_yticks([])
    att = attax.imshow(attr0,interpolation=None)
    attax.spines["top"].set_visible(True)
    attax.spines["right"].set_visible(True)

    # Rewards
    r_max = np.max(crwds)

    rwdax.set_title("Cumulative Reward")
    rwdax.set_xlim([0, t_max])
    rwdax.set_ylim([0, r_max])

    rwdax.plot(trng, crwds, "k-")
    rwdax.vlines(dndices, 0, r_max, linestyle="dashed", linewidth=1, color="blue")
    (rline,) = rwdax.plot(trng[0], crwds[0], "g-", linewidth=1)

   # Health dynamics
    hlthax.set_title("Health")
    hlthax.set_xlim([0, t_max])
    hlthax.set_ylim([0, 100])

    hlthax.plot(trng, hlths, "k-")
    hlthax.vlines(dndices, 0, 100, linestyle="dashed", linewidth=1, color="blue")

    (hline,) = hlthax.plot(trng[0], hlths[0], "r-", linewidth=1)

    if not animate:

        return fig

    else:

        def update(i):

            img = imgs[:, :, :, i]
            im.set_array(img)

            nimg = nimgs[:, :, :, i]
            nim.set_array(nimg)

            attr = attrs[:, :, :, i]
            att.set_array(attr)

            rline.set_data(trng[0:i], crwds[0:i])
            hline.set_data(trng[0:i], hlths[0:i])

        anim = FuncAnimation( fig, update
                             , frames=tqdm( range(1, t_max), desc="Animating Simulation" )
                             , interval=1000 / fps )

        return anim

def plot_acts_tsne_stim(sim_recs): # plot sorted activations

    ltnts = sim_recs["ltnts"]
    hlths = sim_recs["hlths"]

    # zscore
    data=row_zscore(ltnts)
    #data=ltnts
    # get tSNE sorting and resort data
    embedding = fit_tsne_1d(data)
    temp = np.argsort(embedding[:,0])
    data = data[temp,:]
    print(np.amax(data))
    print(np.amin(data))

    # get stimulus collection times
    pos_col = np.where(np.sign(get_stim_coll(hlths)) == 1)
    neg_col = np.where(np.sign(get_stim_coll(hlths)) == -1)

    # plot
    fig = plt.figure(figsize=(10,3), dpi = 400)
    #plt.imshow(data, cmap='bwr', interpolation='nearest', aspect='auto')
    plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto', vmin=-4, vmax=4)
    plt.colorbar()
    plt.vlines(pos_col, 0, data.shape[0], color='grey', linewidth=0.3, linestyle='--')
    plt.vlines(neg_col, 0, data.shape[0], color='black', linewidth=0.3, linestyle=':')
    plt.xlabel('Time (stamps)')
    plt.ylabel(f'unit id.')

    return fig

def receptive_field_plots(rfs):
    """
    Returns a figures of the RGB receptive fields for each element from a layer-wise dictionary.
    """

    figs = {}

    for ky in rfs:

        lyr = rfs[ky]
        ochns,nclrs,_,_ = lyr.shape

        fig, axs = plt.subplots(
            nclrs,ochns,
            figsize=(ochns*1.5, nclrs),
            )

        clrs = ['Red','Green','Blue']
        cmaps = ['inferno', 'viridis', 'cividis']

        for i in range(ochns):

            mx = np.amax(lyr[i])
            mn = np.amin(lyr[i])

            for j in range(nclrs):

                ax = axs[j][i]
                im = ax.imshow(lyr[i,j,:,:],cmap=cmaps[j],vmin=mn,vmax=mx)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)

                if i==0:
                    fig.colorbar(im, ax=ax,cmap=cmaps[j],label=clrs[j],location="left")
                else:
                    fig.colorbar(im, ax=ax,cmap=cmaps[j],location="left")

        figs[ky] = fig

    return figs
