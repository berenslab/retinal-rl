#from pygifsicle import optimize
#import os
#import imageio
#import torch
#from torch.utils import tensorboard
#import wandb
#from sample_factory.utils.utils import AttrDict
#from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
#import pandas as pd
#import numpy as np
#import matplotlib.animation as animation
#
#from retinal_rl.analysis.statistics import spike_triggered_average,fit_tsne_1d,fit_tsne,fit_pca,get_stim_coll,row_zscore

import numpy as np
import matplotlib as mpl
mpl.use("cairo")

import matplotlib.pyplot as plt
plt.style.use('misc/default.mplstyle')

from matplotlib.animation import FuncAnimation

from tqdm.auto import tqdm

def simulation_plot(sim_recs,animate=False,fps=35):

    imgs = sim_recs["imgs"]
    hlths = sim_recs["hlths"]
    img0 = imgs[:, :, :, 0]
    t_max = imgs.shape[3]

    mosaic = """
    aab
    aac
    """

    fig, ax_dict = plt.subplot_mosaic(
        mosaic,
        figsize=(6, 4),
        facecolor="white",
        dpi=200,
        layout="constrained",
    )

    imax = ax_dict["a"]
    clrax = ax_dict["b"]
    hlthax = ax_dict["c"]

    # FoV
    imax.set_title("Field of View")
    imax.set_xticks([])
    imax.set_yticks([])
    im = imax.imshow(img0,interpolation=None)
    imax.spines["top"].set_visible(True)
    imax.spines["right"].set_visible(True)

    # Colour distribution
    clrax.set_title("Colour intensities")
    clrax.set_xlim([0, 255])
    clrax.set_ylim([0, 0.025])

   # Health dynamics
    hlthax.set_title("Health")
    hlthax.set_xlim([0, t_max])
    hlthax.set_ylim([0, 100])
    hrng = np.linspace(0, t_max - 1, t_max)

    hlthax.plot(hrng, hlths, color="grey")

    (hline,) = hlthax.plot(hrng[0], hlths[0], color="red")

    if not animate:

        return fig

    else:

        def update(i):

            img = imgs[:, :, :, i]
            im.set_array(img)

            #rline.set_ydata(rkde(crng))
            #gline.set_ydata(gkde(crng))
            #bline.set_ydata(bkde(crng))

            hline.set_data(hrng[0:i], hlths[0:i])

        anim = FuncAnimation( fig, update
                             , frames=tqdm( range(0, t_max), desc="Animating Simulation" )
                             , interval=1000 / fps )

        return anim
        #anim.save("test.gif")


#def save_simulation_gif(cfg, all_img):
#    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
#    pth = f'{cfg.train_dir}/{cfg.experiment}/sim_{t_stamp}.gif'
#
#    wrt = imageio.get_writer(pth, mode='I',fps=35)
#
#    with wrt as writer:
#        for i in range(all_img.shape[3]):
#            writer.append_data(all_img[:,:,:,i])
#
#    optimize(pth)

    # displaying in WandB
    # if cfg.with_wandb:
    #     wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
    #     wandb.log({"video": wandb.Video(pth, fps=35, format="gif")})

#def save_receptive_fields_plot(cfg,device,enc,lay,env):
#
#    obs = env.reset() # this is the first observation when agent is spawned
#    obs_torch = AttrDict(transform_dict_observations(obs))
#    for key, x in obs_torch.items():
#        obs_torch[key] = torch.from_numpy(x).to(device).float()
#
#    if cfg.greyscale:
#        obs = obs_torch['obs'][:,0,None,:,:] # setting shape of observation to be only single channel
#    else:
#        obs = obs_torch['obs']
#
#    isz = list(obs.size())[1:]
#    outmtx = enc.conv_head[0:(lay*2)](obs)
#    osz = list(outmtx.size())[1:]
#
#    nchns = isz[0]
#    flts = osz[0]
#
#    if cfg.retinal_stride == 2 and cfg.kernel_size == 9: # lindsay with stride and kernel 9
#        diams = [1, 9, 17, 50]
#        rds = diams[lay]//2 + 2*lay# padding a bit
#    else: # classic lindsay case
#        rds = 2 + (8*(lay-1) + enc.kernel_size) // 2
#
#    rwsmlt = 2 if flts > 8 else 1 # rows in rf subplot
#    fltsdv = flts//rwsmlt
#
#    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
#    csv_dir = pth_csv = cfg.train_dir +  "/" + cfg.experiment + f"/rfs-{t_stamp}/"
#
#    if not os.path.exists(csv_dir):
#        os.mkdir(csv_dir)
#
#    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv,dpi = 100,figsize=(6*fltsdv, 4*rwsmlt*nchns))
#
#    for i in range(fltsdv):
#
#        for j in range(rwsmlt):
#
#            flt = i + j*fltsdv
#            avg = spike_triggered_average(device,enc,lay,flt,rds,isz)
#
#            vmx = abs(avg).max()
#
#            for k in range(nchns):
#
#                # Plotting statistics
#                rw = k + j*nchns
#
#                if flts > 1 and nchns*rwsmlt > 1:
#                    ax = axs[rw,i]
#                elif flts <= 1 and nchns*rwsmlt > 1:
#                    ax = axs[rw]
#                elif flts > 1 and nchns*rwsmlt <= 1:
#                    ax = axs[i]
#                else:
#                    ax=axs
#
#                ax.set_axis_off()
#                mtx = avg[k,:,:]
#                pnl = ax.imshow(mtx,vmin=-vmx,vmax=vmx)
#                cbar = fig.colorbar(pnl, ax=ax)
#                cbar.ax.tick_params(labelsize=12*rwsmlt)
#
#                if k == 0:
#                    ax.set_title("Filter: " + str(flt), { 'weight' : 'bold' }, fontsize=12*rwsmlt)
#
#                #rf_dict[f'lay{lay}_flt{flt}_pixch{k}'] = avg[k,:,:].tolist()
#
#                pth_csv = csv_dir + f"rf_lay{lay}_flt{flt}_pixch{k}.csv"
#                mtx_df = pd.DataFrame(mtx)
#                mtx_df.to_csv(pth_csv,header=False,index=False)
#                #np.save(pth_csv, rf_dict, allow_pickle=True)
#
#    # displaying in tensorboard
#    pth_tb = cfg.train_dir +  "/" + cfg.experiment + '/.summary/0/'
#    writer = tensorboard.SummaryWriter(pth_tb)
#    writer.add_figure(f"rf-conv{lay}", fig, close=False) # only show the latest rfs in tb (for comparison across models)
#    writer.close()
#
#    # displaying in WandB
#    if cfg.with_wandb:
#        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
#        wandb.log({f"rf-conv{lay}": fig})
#
#    # saving
#    pth = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{lay}_" + t_stamp + ".png"
#    plt.savefig(pth)
#
#
#    # csv file
#    # pth_csv = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{lay}.csv"
#    # with open('pth_csv', 'w') as csvfile:
#    #     writer = csv.DictWriter(csvfile, fieldnames = employee_info)
#    #     writer.writeheader()
#    #     writer.writerows(new_dict)
#
#def plot_all_rf(cfg, actor_critic, env):
#    enc = actor_critic.encoder.base_encoder
#    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda') # this line should go above the get_acts_actor function
#
#    # always plot receptive fields
#    n_conv_lay = len(enc.conv_head)//2
#    for lay in range(1, n_conv_lay+1):
#        save_receptive_fields_plot(cfg,device,enc,lay,env)
#
## for simulated experience
#def plot_acts_tsne_stim(cfg, acts, health, title): # plot sorted activations
#
#    # zscore
#    data=row_zscore(acts)
#    #data=acts
#    # get tSNE sorting and resort data
#    embedding = fit_tsne_1d(data)
#    temp = np.argsort(embedding[:,0])
#    data = data[temp,:]
#
#    # get stimulus collection times
#    pos_col = np.where(np.sign(get_stim_coll(health)) == 1)
#    neg_col = np.where(np.sign(get_stim_coll(health)) == -1)
#
#    # plot
#    fig = plt.figure(figsize=(10,3), dpi = 400)
#    #plt.imshow(data, cmap='bwr', interpolation='nearest', aspect='auto')
#    plt.imshow(data, cmap='bwr', interpolation='nearest', aspect='auto', vmin=-4, vmax=4)
#    plt.colorbar()
#    plt.vlines(pos_col, 0, data.shape[0], color='grey', linewidth=0.3, linestyle='--')
#    plt.vlines(neg_col, 0, data.shape[0], color='black', linewidth=0.3, linestyle=':')
#    plt.xlabel('Time (stamps)')
#    plt.ylabel(f'{title} unit id.')
#    plt.title(f'Activations of {title} neurons')
#
#    # displaying in WandB
#    if cfg.with_wandb:
#        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
#        wandb.log({f"acts_sorted_tsne_{title}_sim": fig})
#
#    # saving
#    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
#    pth = cfg.train_dir +  "/" + cfg.experiment + f"/acts_sorted_tsne_{title}_sim_" + t_stamp + ".png"
#    plt.savefig(pth)
#
## for dataset inputs
#def plot_dimred(cfg, embedding, c, title='embedding'):
#    plt.figure(figsize=(20,20))
#    plt.scatter(embedding[:,0],embedding[:,1], s=5, c=c, cmap='jet')
#    plt.colorbar()
#    plt.axis('off')
#
#    # saving
#    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
#    pth = cfg.train_dir +  "/" + cfg.experiment + f"/acts_{title}_" + t_stamp + ".png"
#    plt.savefig(pth)
#
#    # displaying in WandB
#    if cfg.with_wandb:
#        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
#        wandb.log({f"acts_{title}": wandb.Image(pth)})
#
#def plot_dimred_ds_acts(cfg, data, all_lab):
#
#    tsne_emb = fit_tsne(data)
#    plot_dimred(cfg, tsne_emb, all_lab, title='tsne_FC_ds')
#
#    pca_emb,_ = fit_pca(data)
#    plot_dimred(cfg, pca_emb, all_lab, title='pca_FC_ds')
#
## for simulation
#def plot_dimred_sim_acts(cfg, data, title=''):
#
#    t = np.arange(data.shape[1])
#
#    tsne_emb = fit_tsne(data)
#    plot_dimred(cfg, tsne_emb, t, title=f'tsne_{title}_sim')
#
#    pca_emb,_ = fit_pca(data)
#    plot_dimred(cfg, pca_emb, t, title=f'pca_{title}_sim')
#
#def save_activations_gif(cfg, imgs, conv_acts, lay, vscale=100):
#
#    snapshots = np.asarray(conv_acts[lay])
#    fps = 30
#    nSeconds = round(len(snapshots)//fps)
#
#    nflts = snapshots[0].shape[2]
#    rwsmlt = 2 if nflts > 8 else 1 # number of rows
#    fltsdv = nflts//rwsmlt + 1 # numer of clumns (+ 1 for rgb image)
#
#    fig, axs = plt.subplots(rwsmlt,fltsdv,dpi=1, figsize=(cfg.res_w*fltsdv, cfg.res_h*rwsmlt))
#
#    ims = []
#    vmaxs = [abs(snapshots[:,:,:,i]).max() for i in range(nflts)]
#
#    print(f'Visualising activations for conv{lay+1}')
#    for t in range(fps*nSeconds):
#        pts = []
#        ax = axs[0,0] if rwsmlt>1 else axs[0]
#        im = ax.imshow(imgs[:,:,:,t], interpolation='none', aspect='auto', vmin=-vscale, vmax=vscale)
#        ax.axis('off')
#        ax = axs[1,0] if rwsmlt>1 else axs[0]
#        ax.axis('off')
#        pts.append(im)
#
#        flt=0 # counter
#        for i in range(fltsdv-1):
#            for j in range(rwsmlt):
#
#                ax = axs[j, i+1] if rwsmlt>1 else axs[i+1]
#                im = ax.imshow(snapshots[t,:,:,flt], interpolation='none', aspect='auto', cmap='bwr', vmin=-vmaxs[j], vmax=vmaxs[j]) # plotting activations
#                flt +=1
#                ax.axis('off')
#                pts.append(im)
#        ims.append(pts)
#        if t % (fps*10) == 0:
#            print('Progress: ', t//fps, '/', nSeconds, ' seconds')
#    anim = animation.ArtistAnimation(fig, ims, interval=1000/fps)
#    print('Saving video... (this can take some time for >10s simulations)')
#
#    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
#    pth = cfg.train_dir + "/" + cfg.experiment + f"/act-conv{lay+1}_" + t_stamp + ".gif"
#
#
#    anim.save(pth, fps=fps)
#
#    # displaying in WandB
#    if cfg.with_wandb:
#        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
#        wandb.log({"video": wandb.Video(pth, fps=35, format="gif")})
#
#    print('Done!')
#
### from here on are post-Cosyne functions/classes (action-analysis and explainability)
#
#def plot_fov_att_overlay_avg(att_out, in_image):
#    plt.plot(dpi=200)
#
#    att_out_i = torch.mean(att_out,0)
#    vlim = np.max(np.abs(att_out_i.cpu().numpy()))
#
#    plt.title('avg')
#    plt.axis('off')
#    plt.imshow(in_image.detach().numpy().transpose(1,2,0)/256, alpha=0.5)
#    plt.imshow(att_out_i.cpu(), alpha=0.5, cmap='seismic', vmin=-vlim, vmax=vlim)
#
#    plt.show()
#
#def plot_fov_att_overlay_rgb(att_out, in_image):
#    fig, axs = plt.subplots(1,3, dpi=200)
#
#    for i in range(3): # rgb
#        att_out_i = att_out[i,:,:]
#        vlim = np.max(np.abs(att_out_i.cpu().numpy()))
#
#        ch = ('R','G','B')[i]
#        axs[i].set_title(ch)
#        axs[i].axis('off')
#        axs[i].imshow(in_image.detach().numpy().transpose(1,2,0)/256, alpha=0.5)
#        axs[i].imshow(att_out_i.cpu(), alpha=0.5, cmap='seismic', vmin=-vlim, vmax=vlim)
#
#    plt.show()
#
#def plot_fov_att_overlay_avg(att_out, in_image, ax):
#    plt.plot(dpi=200)
#
#    att_out_i = torch.mean(att_out,0)
#    vlim = np.max(np.abs(att_out_i.cpu().numpy()))
#
#    ax.axis('off')
#    ax.imshow(in_image.detach().numpy().transpose(1,2,0)/256, alpha=0.5)
#    ax.imshow(att_out_i.cpu(), alpha=0.5, cmap='seismic', vmin=-vlim, vmax=vlim)
#
#def plot_att_action(padded_ds, att_method, action_idx):
#
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#    rows = 5
#    cols = 5
#
#    fig, axs = plt.subplots(rows, cols, dpi=1000, figsize=(cols, rows*(3/4)))
#
#    count = 0
#    for i in range(rows):
#        for j in range (cols):
#            in_image = padded_ds[count][0]
#            attribution = att_method.attribute(in_image.to(device), target=action_idx)
#            # plot_fov_att_overlay_rgb(attribution_gbp, in_image)
#            plot_fov_att_overlay_avg(attribution, in_image, axs[i, j])
#            count += 1
#
#    plt.suptitle(f'Attribution for action index: {action_idx}', fontsize=7)
#    plt.show()
#
#def plot_example_stim_torch(stim):
#    plt.figure(dpi=50)
#    plt.imshow(stim.detach().numpy().transpose(1,2,0)/256)
#    plt.axis('off')
#    plt.title('example stimulus')
#    plt.show()
#
#def plot_violin_action(all_action, all_label):
#
#    fig, axs = plt.subplots(4,1,figsize=(10,8), dpi=200)
#
#    for idx_label in range(10):
#        f = axs[0].violinplot(all_action[4,all_label==idx_label], positions=[idx_label], showmeans=False, showmedians=True, showextrema=False)
#        b = axs[1].violinplot(all_action[5,all_label==idx_label], positions=[idx_label], showmeans=False, showmedians=True, showextrema=False)
#        l = axs[2].violinplot(all_action[1,all_label==idx_label], positions=[idx_label], showmeans=False, showmedians=True, showextrema=False)
#        r = axs[3].violinplot(all_action[2,all_label==idx_label], positions=[idx_label], showmeans=False, showmedians=True, showextrema=False)
#
#        for vplot in [f, b, l, r]: # color coding
#            vplot['cmedians'].set_color('red') if idx_label < 5 else vplot['cmedians'].set_color('green')
#            for pc in vplot['bodies']:
#                pc.set_facecolor('red') if idx_label < 5 else pc.set_facecolor('green')
#                pc.set_edgecolor('black')
#
#
#        axs[0].set_title('Forward')
#        axs[1].set_title('Backward')
#        axs[2].set_title('Left')
#        axs[3].set_title('Right')
#
#        axs[3].set_xlabel('Stimulus ID')
#        axs[0].set_ylabel('Probability')
#
#        xticks = np.arange(0,10)
#        axs[0].set_xticks([])
#        axs[1].set_xticks([])
#        axs[2].set_xticks([])
#        axs[3].set_xticks(xticks)
#
#        axs[0].set_ylim((-0.1,1.1))
#        axs[1].set_ylim((-0.1,1.1))
#        axs[2].set_ylim((-0.1,1.1))
#        axs[3].set_ylim((-0.1,1.1))
#
#    plt.suptitle('P(action|stim_category)', size=20)
#
#    plt.show()
#
#
#def get_vlim_or_all_im(cfg, analyze_out, att_method, target, fixed_vlim=False, vlim_value=None):
#
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#    all_im = []
#    all_vlim = [] # hack to not throw error in case of fixed vlim
#    fig, ax = plt.subplots(1,1,dpi=4, figsize=(cfg.res_w, cfg.res_h))
#
#    for i in range(0,1000,1):
#        in_image = analyze_out['all_img'][:,:,:,i].transpose(2, 0, 1)
#        in_image = torch.tensor(in_image).float()
#
#        attribution = att_method.attribute(in_image.to(device), target=target)
#
#        att_out = attribution
#
#        att_out_i = torch.mean(att_out,0)
#
#        if not fixed_vlim:
#            vlim = np.max(np.abs(att_out_i.cpu().numpy()))
#            all_vlim.append(vlim)
#
#        elif fixed_vlim:
#            vlim = vlim_value
#            plt.plot(dpi=200)
#            plt.axis('off')
#            im1 = ax.imshow(att_out_i.cpu(), alpha=0.8, cmap='seismic', vmin=-vlim, vmax=vlim)
#            im2 = ax.imshow(in_image.detach().numpy().transpose(1,2,0)/256, alpha=0.2)
#            all_im.append([im1, im2])
#    # get max vlim across all frames
#    if not fixed_vlim:
#        max_vlim = np.max(all_vlim)
#
#    return max_vlim if not fixed_vlim else (all_im, fig)
