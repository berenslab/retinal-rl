# Science
from functools import cmp_to_key
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
from pygifsicle import optimize
from torch.utils.tensorboard import SummaryWriter

from openTSNE import TSNE

#from torchinfo import summary
#from captum.attr import NeuronGradient


def save_simulation_gif(cfg,imgs, lay=0):
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir + "/" + cfg.experiment + f"/sim_" + t_stamp + ".gif"

    wrt = imageio.get_writer(pth, mode='I',fps=35)

    with wrt as writer:
        for img in imgs:
            writer.append_data(img)

    optimize(pth)

def save_receptive_fields_plot(cfg,device,enc,lay,obs_torch):
    isz = list(obs_torch['obs'].size())[1:]
    outmtx = enc.conv_head[0:(lay*2)](obs_torch['obs'])
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

    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv,dpi = 100,figsize=(6*fltsdv, 4*rwsmlt*nchns))

    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,lay,flt,rds,isz)

            vmx = abs(avg).max()

            for k in range(nchns):

                # Plotting statistics
                rw = k + j*nchns
                ax = axs[rw,i] if flts > 1 else axs[rw]
                ax.set_axis_off()
                pnl = ax.imshow(avg[k,:,:],vmin=-vmx,vmax=vmx)
                cbar = fig.colorbar(pnl, ax=ax)
                cbar.ax.tick_params(labelsize=12*rwsmlt)

                if k == 0:
                    ax.set_title("Filter: " + str(flt), { 'weight' : 'bold' }, fontsize=12*rwsmlt)
    

    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')

    # displaying in tensorboard
    pth_tb = cfg.train_dir +  "/" + cfg.experiment + '/.summary/0/'
    writer = torch.utils.tensorboard.SummaryWriter(pth_tb)
    writer.add_figure(f"rf-conv{lay}", fig, close=False) # only show the latest rfs in tb (for comparison across models)
    writer.close()

    # saving
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{lay}_" + t_stamp + ".png"
    plt.savefig(pth)  


def spike_triggered_average(dev,enc,lay,flt,rds,isz):

    with torch.no_grad():

        btchsz = [50000] + isz
        cnty = (1+btchsz[2])//2
        cntx = (1+btchsz[3])//2
        mny = cnty - rds
        mxy = cnty + rds
        mnx = cntx - rds
        mxx = cntx + rds
        obsns = torch.randn(size=btchsz,device=dev)
        outmtx = enc.conv_head[0:(lay*2)](obsns) #forward pass
        outsz = outmtx.size()
        outs = outmtx[:,flt,outsz[2]//2,outsz[3]//2].cpu()
        obsns1 = obsns[:,:,mny:mxy,mnx:mxx].cpu()
        avg = np.average(obsns1,axis=0,weights=outs)

    return avg

def save_activations_gif(cfg, imgs, conv_acts, lay): 
    
    snapshots = np.asarray(conv_acts)
    fps = 30
    nSeconds = round(len(snapshots)//fps)
    
    nflts = snapshots[0].shape[2]
    rwsmlt = 2 if nflts > 8 else 1 # number of rows 
    fltsdv = nflts//rwsmlt + 1 # numer of clumns (+ 1 for rgb image)
    
    fig, axs = plt.subplots(rwsmlt,fltsdv,dpi=1, figsize=(128*fltsdv, 72*rwsmlt))
    
    ims = []
    vmaxs = [abs(snapshots[:,:,:,i]).max() for i in range(nflts)]
    
    print(f'Visualising activations for conv{lay+1}')

    for t in range(fps*nSeconds-1):
        pts = []

        ax = axs[0,0] if rwsmlt>1 else axs[0]
        im = ax.imshow(imgs[t][:,:,:], interpolation='none', aspect='auto', vmin=-0, vmax=256) # plotting pixels
        ax.axis('off')
        ax = axs[1,0] if rwsmlt>1 else axs[0]
        ax.axis('off')
        pts.append(im)
        
        flt=0 # counter
        for i in range(fltsdv-1):
            for j in range(rwsmlt):
                
                ax = axs[j, i+1] if rwsmlt>1 else axs[i+1]
                im = ax.imshow(snapshots[t,:,:,flt], interpolation='none', aspect='auto', vmin=0, vmax=vmaxs[flt]) # plotting activations
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

    print('Done!')

     
def plot_PCA(cfg, imgs, env_infos, fc_acts, v_acts, n_pcs=64):
    rm_env_steps = 500
    print(f'Removing first {rm_env_steps} to avoid value artefact due to rnn')
    fc_acts_np = np.concatenate(fc_acts[rm_env_steps:], axis=0) # converting from list to np
    v_acts_np = np.concatenate(v_acts[rm_env_steps:])
    env_infos = env_infos[rm_env_steps:]

    # preprocessing/reformatting health, value and pixels
    health = np.zeros(len(env_infos))
    pix_vect = np.zeros((len(env_infos), imgs[0].flatten().shape[0]))

    for i in range(len(health)):
        health[i] = env_infos[i]['HEALTH']
        pix_vect[i,:] = imgs[i].flatten()
    
    # centering and normalising
    health_cent = (health - np.mean(health, axis=0))/np.max(health)


    # fc activation pca
    pca = torch.pca_lowrank(torch.from_numpy(fc_acts_np), q=n_pcs)
    var_exp = pca[1].numpy()/np.sum(pca[1].numpy())
    ll_states = pca[0].numpy()[:,0:n_pcs]
    t_stamps =np.arange(0, len(ll_states))
    
    # pixel pca
    pix_pca = torch.pca_lowrank(torch.from_numpy(pix_vect), q=n_pcs)
    pix_var_exp = pix_pca[1].numpy()/np.sum(pix_pca[1].numpy())
    pix_ll_states = pix_pca[0].numpy()[:,0:n_pcs]
    
    health = health_cent*2*np.max(ll_states[:,0]) + np.mean(ll_states[:,0]) #rescale and center to PC1 over time

    plt.subplots(2, 2, gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [5, 1]}, figsize=(20,10))
    
    # plot of first two PCs
    plt.subplot(2, 2, 1)
    plt.scatter(ll_states[:,0], ll_states[:,1], s=0.1, c=v_acts_np, cmap='jet')
    plt.colorbar(label="Value")
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    
    #scree plot
    plt.subplot(2, 2, 2)
    plt.plot(var_exp, label='PCA on output')
    plt.plot(pix_var_exp, label='PCA on pixels')
    plt.ylim(ymin=0)
    plt.ylabel('Proportion var. explained')
    plt.xlabel('PC number')

    # first pc as time series
    plt.subplot(2, 2, (3,4))
    plt.plot(t_stamps/1000, normalize_data(ll_states[:,0]), label='PC1 output')
    plt.plot(t_stamps/1000, normalize_data(pix_ll_states[:,0]), label='PC1 pixels')
    plt.plot(t_stamps/1000, normalize_data(health), label='health', c='grey')
    plt.plot(t_stamps/1000, normalize_data(v_acts_np), label='value', c='red')
    plt.xlabel('Time(kStamps)')
    plt.ylabel('PC1')
    plt.legend()
    
    # saving
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + "/fc_pca_" + t_stamp + ".png"
    plt.savefig(pth)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot_tsne(cfg, all_acts_dict):
    nn_acts_plot = ['fc_acts', 'rnn_acts'] # hard-coded which activations to plot
    
    _, axs = plt.subplots(1,len(nn_acts_plot),figsize=(40,10))

    for (i, nn_acts) in enumerate(nn_acts_plot):
        print('\n Running tSNE for: ' + nn_acts)
        plot_lay_tsne(all_acts_dict[nn_acts], all_acts_dict['v_acts'], axs[i])
        axs[i].set_title(nn_acts)
    
    # saving
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + "/tsne_env_" + t_stamp + ".png"
    plt.savefig(pth)

def plot_lay_tsne(nn_acts, v_acts, ax):
    rm_env_steps = 500
    print(f'Removing first {rm_env_steps} to avoid value artefact due to rnn')
    nn_acts_np = np.concatenate(nn_acts[rm_env_steps:], axis=0)
    v_acts_np = np.concatenate(v_acts[rm_env_steps:])

    tsne = TSNE(
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )

    embedding = tsne.fit(nn_acts_np)

    #plt.figure(figsize=(10,10))
    sct = ax.scatter(embedding[:,0], embedding[:,1], s=0.1, c=v_acts_np,cmap='jet')
    cbar = plt.colorbar(sct, ax=ax)
    cbar.set_label(label='Value',size=20)
    cbar.ax.tick_params(labelsize=50)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    ax.set_axis_off()

def plot_dimred_ds(cfg, nn_acts_np, all_lab_np): 
    
    # computing PCA
    n_pcs=64
    pca = torch.pca_lowrank(torch.from_numpy(nn_acts_np), q=n_pcs)
    ll_states = pca[0].numpy()[:,0:n_pcs]
   
    # computing tSNE
    tsne = TSNE(
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )

    embedding = tsne.fit(nn_acts_np)
    
    # plotting both
    fig, axs = plt.subplots(1, 2, figsize=(20,10))
    sct1 = axs[0].scatter(ll_states[:,0], ll_states[:,1], s=0.6, c=all_lab_np,  cmap='jet')
    axs[0].set_title('PCA')
    cbar = plt.colorbar(sct1, ax=axs[0])
    cbar.set_label(label='Stim id',size=20)
    cbar.outline.set_visible(False)
    axs[0].set_axis_off()

    sct2 = axs[1].scatter(embedding[:,0], embedding[:,1], s=0.1, c=all_lab_np, cmap='jet')#, c=v_acts_np,cmap='jet') # AS A SANITY CHECK ALSO COLOR CODE BY TIME (SHOULD MATCH PROBABLY)
    cbar = plt.colorbar(sct2, ax=axs[1])
    axs[1].set_title('tSNE')
    cbar.set_label(label='Stim id',size=20)
    cbar.outline.set_visible(False)
    axs[1].set_axis_off()
    
    # saving
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + "/tsne_ds_" + t_stamp + ".png"
    plt.savefig(pth)