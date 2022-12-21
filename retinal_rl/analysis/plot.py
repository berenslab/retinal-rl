from pygifsicle import optimize
import os
import imageio
import torch
from torch.utils import tensorboard
import wandb
from sample_factory.utils.utils import AttrDict
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from retinal_rl.analysis.statistics import spike_triggered_average,fit_tsne_1d,fit_tsne,fit_pca,get_stim_coll,row_zscore


def save_simulation_gif(cfg, all_img):
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = f'{cfg.train_dir}/{cfg.experiment}/sim_{t_stamp}.gif'

    wrt = imageio.get_writer(pth, mode='I',fps=35)

    with wrt as writer:
        for i in range(all_img.shape[3]):
            writer.append_data(all_img[:,:,:,i])

    optimize(pth)

    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
        wandb.log({"video": wandb.Video(pth, fps=35, format="gif")})

def save_receptive_fields_plot(cfg,device,enc,lay,env):

    obs = env.reset() # this is the first observation when agent is spawned
    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
        obs_torch[key] = torch.from_numpy(x).to(device).float()

    if cfg.greyscale:
        obs = obs_torch['obs'][:,0,None,:,:] # setting shape of observation to be only single channel
    else:
        obs = obs_torch['obs']

    isz = list(obs.size())[1:]
    outmtx = enc.conv_head[0:(lay*2)](obs)
    osz = list(outmtx.size())[1:]

    nchns = isz[0]
    flts = osz[0]

    if cfg.retinal_stride == 2 and cfg.kernel_size == 9: # lindsay with stride and kernel 9
        diams = [1, 9, 17, 50]
        rds = diams[lay]//2 + 2*lay# padding a bit
    else: # classic lindsay case
        rds = 2 + (8*(lay-1) + enc.kernel_size) // 2

    rwsmlt = 2 if flts > 8 else 1 # rows in rf subplot
    fltsdv = flts//rwsmlt

    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    csv_dir = pth_csv = cfg.train_dir +  "/" + cfg.experiment + f"/rfs-{t_stamp}/"

    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv,dpi = 100,figsize=(6*fltsdv, 4*rwsmlt*nchns))

    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,lay,flt,rds,isz)

            vmx = abs(avg).max()

            for k in range(nchns):

                # Plotting statistics
                rw = k + j*nchns

                if flts > 1 and nchns*rwsmlt > 1:
                    ax = axs[rw,i]
                elif flts <= 1 and nchns*rwsmlt > 1:
                    ax = axs[rw]
                elif flts > 1 and nchns*rwsmlt <= 1:
                    ax = axs[i]
                else:
                    ax=axs

                ax.set_axis_off()
                mtx = avg[k,:,:]
                pnl = ax.imshow(mtx,vmin=-vmx,vmax=vmx)
                cbar = fig.colorbar(pnl, ax=ax)
                cbar.ax.tick_params(labelsize=12*rwsmlt)

                if k == 0:
                    ax.set_title("Filter: " + str(flt), { 'weight' : 'bold' }, fontsize=12*rwsmlt)

                #rf_dict[f'lay{lay}_flt{flt}_pixch{k}'] = avg[k,:,:].tolist()

                pth_csv = csv_dir + f"rf_lay{lay}_flt{flt}_pixch{k}.csv"
                mtx_df = pd.DataFrame(mtx)
                mtx_df.to_csv(pth_csv,header=False,index=False)
                #np.save(pth_csv, rf_dict, allow_pickle=True)

    # displaying in tensorboard
    pth_tb = cfg.train_dir +  "/" + cfg.experiment + '/.summary/0/'
    writer = tensorboard.SummaryWriter(pth_tb)
    writer.add_figure(f"rf-conv{lay}", fig, close=False) # only show the latest rfs in tb (for comparison across models)
    writer.close()

    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
        wandb.log({f"rf-conv{lay}": fig})

    # saving
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{lay}_" + t_stamp + ".png"
    plt.savefig(pth)


    # csv file
    # pth_csv = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{lay}.csv"
    # with open('pth_csv', 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames = employee_info)
    #     writer.writeheader()
    #     writer.writerows(new_dict)

def plot_all_rf(cfg, actor_critic, env):
    enc = actor_critic.encoder.base_encoder
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda') # this line should go above the get_acts_actor function

    # always plot receptive fields
    n_conv_lay = len(enc.conv_head)//2
    for lay in range(1, n_conv_lay+1):
        save_receptive_fields_plot(cfg,device,enc,lay,env)

# for simulated experience
def plot_acts_tsne_stim(cfg, acts, health, title): # plot sorted activations

    # zscore
    data=row_zscore(acts)
    #data=acts
    # get tSNE sorting and resort data
    embedding = fit_tsne_1d(data)
    temp = np.argsort(embedding[:,0])
    data = data[temp,:]

    # get stimulus collection times
    pos_col = np.where(np.sign(get_stim_coll(health)) == 1)
    neg_col = np.where(np.sign(get_stim_coll(health)) == -1)

    # plot
    fig = plt.figure(figsize=(10,3), dpi = 400)
    #plt.imshow(data, cmap='bwr', interpolation='nearest', aspect='auto')
    plt.imshow(data, cmap='bwr', interpolation='nearest', aspect='auto', vmin=-4, vmax=4)
    plt.colorbar()
    plt.vlines(pos_col, 0, data.shape[0], color='grey', linewidth=0.3, linestyle='--')
    plt.vlines(neg_col, 0, data.shape[0], color='black', linewidth=0.3, linestyle=':')
    plt.xlabel('Time (stamps)')
    plt.ylabel(f'{title} unit id.')
    plt.title(f'Activations of {title} neurons')

    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
        wandb.log({f"acts_sorted_tsne_{title}_sim": fig})

    # saving
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/acts_sorted_tsne_{title}_sim_" + t_stamp + ".png"
    plt.savefig(pth)

# for dataset inputs
def plot_dimred(cfg, embedding, c, title='embedding'):
    plt.figure(figsize=(20,20))
    plt.scatter(embedding[:,0],embedding[:,1], s=5, c=c, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    # saving
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/acts_{title}_" + t_stamp + ".png"
    plt.savefig(pth)

    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
        wandb.log({f"acts_{title}": wandb.Image(pth)})

def plot_dimred_ds_acts(cfg, data, all_lab):

    tsne_emb = fit_tsne(data)
    plot_dimred(cfg, tsne_emb, all_lab, title='tsne_FC_ds')

    pca_emb,_ = fit_pca(data)
    plot_dimred(cfg, pca_emb, all_lab, title='pca_FC_ds')

# for simulation
def plot_dimred_sim_acts(cfg, data, title=''):

    t = np.arange(data.shape[1])

    tsne_emb = fit_tsne(data)
    plot_dimred(cfg, tsne_emb, t, title=f'tsne_{title}_sim')

    pca_emb,_ = fit_pca(data)
    plot_dimred(cfg, pca_emb, t, title=f'pca_{title}_sim')

def save_activations_gif(cfg, imgs, conv_acts, lay, vscale=100):

    snapshots = np.asarray(conv_acts[lay])
    fps = 30
    nSeconds = round(len(snapshots)//fps)

    nflts = snapshots[0].shape[2]
    rwsmlt = 2 if nflts > 8 else 1 # number of rows
    fltsdv = nflts//rwsmlt + 1 # numer of clumns (+ 1 for rgb image)

    fig, axs = plt.subplots(rwsmlt,fltsdv,dpi=1, figsize=(cfg.res_w*fltsdv, cfg.res_h*rwsmlt))

    ims = []
    vmaxs = [abs(snapshots[:,:,:,i]).max() for i in range(nflts)]

    print(f'Visualising activations for conv{lay+1}')
    for t in range(fps*nSeconds):
        pts = []
        ax = axs[0,0] if rwsmlt>1 else axs[0]
        im = ax.imshow(imgs[:,:,:,t], interpolation='none', aspect='auto', vmin=-vscale, vmax=vscale)
        ax.axis('off')
        ax = axs[1,0] if rwsmlt>1 else axs[0]
        ax.axis('off')
        pts.append(im)

        flt=0 # counter
        for i in range(fltsdv-1):
            for j in range(rwsmlt):

                ax = axs[j, i+1] if rwsmlt>1 else axs[i+1]
                im = ax.imshow(snapshots[t,:,:,flt], interpolation='none', aspect='auto', cmap='bwr', vmin=-vmaxs[j], vmax=vmaxs[j]) # plotting activations
                flt +=1
                ax.axis('off')
                pts.append(im)
        ims.append(pts)
        if t % (fps*10) == 0:
            print('Progress: ', t//fps, '/', nSeconds, ' seconds')
    anim = animation.ArtistAnimation(fig, ims, interval=1000/fps)
    print('Saving video... (this can take some time for >10s simulations)')

    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir + "/" + cfg.experiment + f"/act-conv{lay+1}_" + t_stamp + ".gif"


    anim.save(pth, fps=fps)

    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project=cfg.wandb_project, group='rfs')
        wandb.log({"video": wandb.Video(pth, fps=35, format="gif")})

    print('Done!')
