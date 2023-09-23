import numpy as np
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
plt.style.use('misc/default.mplstyle')

import seaborn as sns

from PIL import Image
from torchvision.transforms.functional import adjust_contrast
from matplotlib.animation import FuncAnimation, AbstractMovieWriter
from retinal_rl.analysis.statistics import fit_tsne_1d,get_stim_coll,row_zscore

from tqdm.auto import tqdm

greyscale = np.array([0.299, 0.587, 0.114])

# Custom writer class to save frames as PNG files
class PNGWriter(AbstractMovieWriter):
    def setup(self, fig, outfile, dpi, *args):
        self.outfile = outfile
        self.dpi = dpi
        self._frame_counter = 0  # Initialize frame counter

    def grab_frame(self, **savefig_kwargs):
        plt.savefig(f"{self.outfile}/frame_{self._frame_counter}.png", format='png')
        self._frame_counter += 1  # Increment frame counter

    def finish(self):
        pass  # No action needed for PNGs

    @classmethod
    def isAvailable(cls):
        return True

def enhance_attribution(attr_series):
    lower_bound = np.percentile(attr_series, 0)
    upper_bound = np.percentile(attr_series, 99.9)
    enhanced_series = np.clip(attr_series, lower_bound, upper_bound)
    enhanced_series_normalized = ((enhanced_series - enhanced_series.min()) / 
                                  (enhanced_series.max() - enhanced_series.min()) * 255).astype(np.uint8)
    return enhanced_series_normalized


def simulation_plot(sim_recs,frame_step=None, val_mu=None, val_sigma=None, animate=False,fps=35,prgrs=True):

    imgs = sim_recs["imgs"]
    hlths0 = sim_recs["hlths"]
    vals0 = sim_recs["vals"]
    ltnts = sim_recs["ltnts"]
    dns = sim_recs["dns"]
    plcys = sim_recs["plcys"].detach().cpu().numpy()

    if val_mu is None:
        val_mu = np.mean(vals0)
    if val_sigma is None:
        val_sigma = np.std(vals0)

    attrs = enhance_attribution(sim_recs["attrs"])
    dndices = np.where(dns)

    vals = np.ma.array(vals0, mask=dns)
    hlths = np.ma.array(hlths0, mask=dns)

    t_max = imgs.shape[3]

    # Mosaic layout
    mosaic = """
        aaccc
        aaddd
        bbeee
        bbfff
        """

    fig, ax_dict = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 7),
        constrained_layout=True,
    )

    imax = ax_dict["a"]
    attax = ax_dict["b"]
    satax = ax_dict["c"]
    ltntax = ax_dict["d"]
    action1_ax = ax_dict["e"]
    action2_ax = ax_dict["f"]

    ltntax.sharex(satax)
    action1_ax.sharex(satax)
    action2_ax.sharex(satax)

    # Example using the pastel palette
    colors_pastel = sns.color_palette("pastel", 6)
    
    # Example using the Blues palette for the first three actions and Purples for the next three
    colors_blues_purples = sns.color_palette("Blues", 3) + sns.color_palette("Purples", 3)
    
    # Example using the cubehelix palette
    colors_cubehelix = sns.cubehelix_palette(6, start=.5, rot=-.75)
    
    # Choose one of the above color sets and replace the original 'colors' variable
    colors = colors_blues_purples  # or colors_blues_purples or colors_cubehelix

    # Policy for Action 1
    bottom = np.zeros(t_max)
    hlabels = ["Centre", "Left", "Right"]
    for j in range(3):
        action1_ax.fill_between(range(t_max), bottom, bottom + plcys[0, j, :], color=colors[j], label=hlabels[j])
        bottom += plcys[0, j, :]
    action1_ax.set_title("Heading Distribution")
    action1_ax.set_xlim(0, t_max)
    action1_ax.set_ylim(0, 1)
    action1_ax.set_xlabel('Time (Frames)')
    action1_ax.set_ylabel('Probability')
    action1_ax.set_yticks([0,1])
    action1_ax.legend(loc='upper right')

    vlabels = ["Stationary", "Forward", "Backward"]
    # Policy for Action 2
    bottom = np.zeros(t_max)
    for j in range(3):
        action2_ax.fill_between(range(t_max), bottom, bottom + plcys[1, j, :], color=colors[j+3], label=vlabels[j])
        bottom += plcys[1, j, :]
    action2_ax.set_title("Velocity Distribution")
    action2_ax.set_xlim(0, t_max)
    action2_ax.set_ylim(0, 1)
    action2_ax.set_xlabel('Time (Frames)')
    action2_ax.set_ylabel('Probability')
    action2_ax.legend(loc='upper right')
    action2_ax.set_yticks([0,1])

    # FoV
    imax.set_title("Field of View")
    imax.set_xticks([])
    imax.set_yticks([])
    im = imax.imshow(imgs[:, :, :, 0], interpolation=None)
    imax.spines["top"].set_visible(True)
    imax.spines["right"].set_visible(True)

    # Attribution
    attax.set_title("Attribution")
    attax.set_xticks([])
    attax.set_yticks([])
    attax.spines["top"].set_visible(True)
    attax.spines["right"].set_visible(True)

    attr0 = Image.fromarray(enhance_attribution(attrs[:, :, :, 0]))
    att = attax.imshow(attr0)

    # Health and Rewards
    valax = satax.twinx()

    trng = np.linspace(0, t_max - 1, t_max)

    r_max = np.max(vals)

# Generate y-axis labels based on the mean and standard deviation
    y_labels = [val_mu - 3*val_sigma, val_mu - 2*val_sigma, val_mu - val_sigma, val_mu, val_mu + val_sigma, val_mu + 2*val_sigma, val_mu + 3*val_sigma]
    y_label_strings = ['-3σ', '-2σ', '-σ', 'μ', '+σ', '+2σ', '+3σ']
    valax.set_yticks(y_labels, y_label_strings)  # Set y-axis labels

    valax.set_title("Performance")
    valax.set_xlim([0, t_max])

    valax.plot(trng, vals, "g-")
    valax.vlines(dndices, 0, r_max, linestyle="dashed", linewidth=1, color="blue")
    valax.spines["right"].set_visible(True)

    valax.set_xlabel('Time (Frames)')
    valax.set_ylabel('Value', color='g')

    # Health dynamics
    satax.set_xlim([0, t_max])
    satax.set_ylim([0, 100])

    satax.plot(trng, hlths, "b-")
    satax.vlines(dndices, 0, 100, linestyle="dashed", linewidth=1, color="blue")
    satax.set_ylabel('Satiety', color='b')

    # Latent dynamics
    data = row_zscore(ltnts)
    embedding = fit_tsne_1d(data)
    temp = np.argsort(embedding[:, 0])
    data = data[temp, :]

    pos_col = np.where(np.sign(get_stim_coll(hlths)) == 1)
    neg_col = np.where(np.sign(get_stim_coll(hlths)) == -1)

    ltntax.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
    ltntax.vlines(pos_col, 0, data.shape[0], color='grey', linewidth=0.3, linestyle='--')
    ltntax.vlines(neg_col, 0, data.shape[0], color='black', linewidth=0.3, linestyle=':')
    ltntax.set_xlabel('Time (Frames)')
    ltntax.set_ylabel('Sorted Neuron ID')
    ltntax.set_ylim([0, data.shape[0]])
    ltntax.set_title("Latent State")
    # Show cmap
    # ltntax.figure.colorbar(ltntax.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto', vmin=-1, vmax=1), ax=ltntax)

    # Turn off x-axis labels for the plots "c", "d", and "e"
    for label in ["c", "d", "e"]:
        ax_dict[label].set_xticks([])
        ax_dict[label].set_xlabel('')

    vl1 = valax.axvline(0, color='black', linewidth=1.5, linestyle='-',zorder=10)
    vl2 = ltntax.axvline(0, color='black', linewidth=1.5, linestyle='-',zorder=10)
    vl3 = action1_ax.axvline(0, color='black', linewidth=1.5, linestyle='-',zorder=10)
    vl4 = action2_ax.axvline(0, color='black', linewidth=1.5, linestyle='-',zorder=10)


    # Get the position of the ltntax subplot
    ltntax_pos = ltntax.get_position()

    # Create an axes for the colorbar next to ltntax
    cbar_ax = fig.add_axes([ltntax_pos.x1 + 0.07, ltntax_pos.y0 + 0.025, 0.005, ltntax_pos.height])

    # Add the colorbar to the created axes
    cbar = fig.colorbar(ltntax.images[0], cax=cbar_ax)
    cbar.set_ticks([-1, 0, 1])

    if frame_step is not None:
        im.set_array(imgs[:, :, :, frame_step])
        attr_image = Image.fromarray(enhance_attribution(attrs[:, :, :, frame_step]))
        att.set_array(attr_image)
        vl1.set_xdata([frame_step])
        vl2.set_xdata([frame_step])
        vl3.set_xdata([frame_step])
        vl4.set_xdata([frame_step])

        return fig

    else:
        pbar = tqdm(total=t_max, desc="Animating", ncols=100)

        def update(i):
            img = imgs[:, :, :, i]
            im.set_array(img)

            attr = attrs[:, :, :, i]
            attr_image = Image.fromarray(attr)
            att.set_array(attr_image)

            vl1.set_xdata([i])
            vl2.set_xdata([i])
            vl3.set_xdata([i])
            vl4.set_xdata([i])

            pbar.update(1)

        anim = FuncAnimation(fig, update, frames=range(1, t_max), interval=1000 / 35)  # Assuming 35 FPS
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
    plt.imshow(data, cmap='viridis', interpolation='nearest', aspect='auto', vmin=-4, vmax=4)
    #plt.colorbar()
    plt.vlines(pos_col, 0, data.shape[0], color='grey', linewidth=0.3, linestyle='--')
    plt.vlines(neg_col, 0, data.shape[0], color='black', linewidth=0.3, linestyle=':')
    plt.xlabel('Time (stamps)')
    plt.ylabel(f'unit id.')

    return fig

def receptive_field_plots(lyr):
    """
    Returns a figures of the RGB receptive fields for each element from a layer-wise dictionary.
    """

    ochns,nclrs,_,_ = lyr.shape

    fig, axs0 = plt.subplots(
        nclrs,ochns,
        figsize=(ochns*1.5, nclrs),
        )

    axs = axs0.flat
    clrs = ['Red','Green','Blue']
    cmaps = ['inferno', 'viridis', 'cividis']

    for i in range(ochns):

        mx = np.amax(lyr[i])
        mn = np.amin(lyr[i])

        for j in range(nclrs):

            ax = axs[i + ochns * j]
            #hght,wdth = lyr[i,j,:,:].shape
            im = ax.imshow(lyr[i,j,:,:],cmap=cmaps[j],vmin=mn,vmax=mx)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            if i==0:
                fig.colorbar(im, ax=ax,cmap=cmaps[j],label=clrs[j],location="left")
            else:
                fig.colorbar(im, ax=ax,cmap=cmaps[j],location="left")

    return fig
