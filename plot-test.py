import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from openTSNE import TSNE
import argparse
from matplotlib.animation import FuncAnimation, PillowWriter

def row_zscore(mat):
    return (mat - np.mean(mat,1)[:,np.newaxis])/(np.std(mat,1)[:,np.newaxis]+1e-8)

def fit_tsne_1d(data):
    print('fitting 1d-tSNE...')
    tsne = TSNE(n_components=1, perplexity=20, initialization="pca", metric="euclidean", n_jobs=8, random_state=3)
    tsne_emb = tsne.fit(data)
    return tsne_emb

def get_stim_coll(all_health, health_dep=-8, death_dep=30):
    stim_coll = np.diff(all_health)
    stim_coll[stim_coll == health_dep] = 0
    stim_coll[stim_coll > death_dep] = 0
    stim_coll[stim_coll < -death_dep] = 0
    return stim_coll

def enhance_attribution(attr_series):
    lower_bound = np.percentile(attr_series, 0)
    upper_bound = np.percentile(attr_series, 99.9)
    enhanced_series = np.clip(attr_series, lower_bound, upper_bound)
    enhanced_series_normalized = ((enhanced_series - enhanced_series.min()) / 
                                  (enhanced_series.max() - enhanced_series.min()) * 255).astype(np.uint8)
    return enhanced_series_normalized


def simulation_plot(sim_recs, frame_step=None):

    imgs = sim_recs["imgs"]
    attrs = sim_recs["attrs"]
    hlths0 = sim_recs["hlths"]
    crwds0 = sim_recs["vals"]
    ltnts = sim_recs["ltnts"]
    dns = sim_recs["dns"]
    plcys = sim_recs["plcys"]
    dndices = np.where(dns)

    crwds = np.ma.array(crwds0, mask=dns)
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
        figsize=(10, 8),
    )

    imax = ax_dict["a"]
    attax = ax_dict["b"]
    rwdax = ax_dict["c"]
    ltntax = ax_dict["d"]
    action1_ax = ax_dict["e"]
    action2_ax = ax_dict["f"]

    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

    # Policy for Action 1
    bottom = np.zeros(t_max)
    for j in range(3):
        action1_ax.fill_between(range(t_max), bottom, bottom + plcys[0, j, :], color=colors[j], label=f'Action 1 - Choice {j+1}')
        bottom += plcys[0, j, :]
    action1_ax.set_title("Policy for Action 1")
    action1_ax.set_xlim(0, t_max)
    action1_ax.set_ylim(0, 1)
    action1_ax.set_xlabel('Time (Frames)')
    action1_ax.set_ylabel('Probability')
    action1_ax.legend(loc='upper right')

    # Policy for Action 2
    bottom = np.zeros(t_max)
    for j in range(3):
        action2_ax.fill_between(range(t_max), bottom, bottom + plcys[1, j, :], color=colors[j+3], label=f'Action 2 - Choice {j+1}')
        bottom += plcys[1, j, :]
    action2_ax.set_title("Policy for Action 2")
    action2_ax.set_xlim(0, t_max)
    action2_ax.set_ylim(0, 1)
    action2_ax.set_xlabel('Time (Frames)')
    action2_ax.set_ylabel('Probability')
    action2_ax.legend(loc='upper right')

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
    hlthax = rwdax.twinx()

    trng = np.linspace(0, t_max - 1, t_max)

    r_max = np.max(crwds)

    rwdax.set_title("Reward/Health")
    rwdax.set_xlim([0, t_max])
    rwdax.set_ylim([0, r_max])

    rwdax.plot(trng, crwds, "k-")
    rwdax.vlines(dndices, 0, r_max, linestyle="dashed", linewidth=1, color="blue")
    (rline,) = rwdax.plot(trng[0], crwds[0], "g-", linewidth=1)
    rwdax.spines["right"].set_visible(True)

    rwdax.set_xlabel('Time (Frames)')
    rwdax.set_ylabel('Cumulative Reward')

    # Health dynamics
    hlthax.set_xlim([0, t_max])
    hlthax.set_ylim([0, 100])

    hlthax.plot(trng, hlths, "k-")
    hlthax.vlines(dndices, 0, 100, linestyle="dashed", linewidth=1, color="blue")
    hlthax.set_ylabel('Satiety')

    (hline,) = hlthax.plot(trng[0], hlths[0], "r-", linewidth=1)

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

    vl = ltntax.axvline(0, color='red', linewidth=1, linestyle='-')

    if frame_step is not None:
        im.set_array(imgs[:, :, :, frame_step])
        attr_image = Image.fromarray(enhance_attribution(attrs[:, :, :, frame_step]))
        att.set_array(attr_image)
        rline.set_data(trng[0:frame_step], crwds[0:frame_step])
        hline.set_data(trng[0:frame_step], hlths[0:frame_step])
        vl.set_xdata(frame_step)
        return fig

    else:
        pbar = tqdm(total=t_max, desc="Animating", ncols=100)

        def update(i):
            img = imgs[:, :, :, i]
            im.set_array(img)

            attr = enhanced_attrs[:, :, :, i]
            attr_image = Image.fromarray(attr)
            att.set_array(attr_image)

            rline.set_data(trng[0:i], crwds[0:i])
            hline.set_data(trng[0:i], hlths[0:i])
            vl.set_xdata(i)

            pbar.update(1)

        anim = FuncAnimation(fig, update, frames=range(1, t_max), interval=1000 / 35)  # Assuming 35 FPS
        return anim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize simulation data.")
    parser.add_argument("--frame_step", type=int, default=None, help="Frame step to be visualized. If not provided, animate the entire series.")
    args = parser.parse_args()

    sim_recs = np.load('sim_recs.npy', allow_pickle=True).item()
    enhanced_attrs = enhance_attribution(sim_recs["attrs"])
    result = simulation_plot(sim_recs, frame_step=args.frame_step)

    if args.frame_step is None:
        writer = PillowWriter(fps=35)
        result.save("animation.gif", writer=writer)
    else:
        plt.show()
