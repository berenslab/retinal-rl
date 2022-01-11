# Science
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pygifsicle import optimize

#from torchinfo import summary
#from captum.attr import NeuronGradient


def save_simulation_gif(cfg,imgs):

    pth = cfg.train_dir + "/" + cfg.experiment + "/simulation-" + str(np.datetime64('now')) + ".gif"

    wrt = imageio.get_writer(pth, mode='I',fps=35)

    with wrt as writer:
        for img in imgs:
            writer.append_data(img)

    optimize(pth)

def save_receptive_fields_plot_layerspec(cfg, enc, get_lay=1):
    
    lay_ind = (get_lay-1)*2 # conv(get_lay) layer index in enc.conv_head sequential
    conv_lay = enc.conv_head[lay_ind]

    filt = conv_lay.weight.data.cpu().numpy()
    (nflts, nchns) = filt.shape[0:2]

    fig, axs = plt.subplots(nchns,nflts, figsize=(2*nflts, 2*nchns))

    for j in range(nflts):

        axs_j = axs[:,j] if nflts >1 else axs
        
        for k in range(nchns):
            
            this_filt =  filt[j, k, :, :].squeeze()
            # Plotting statistics
            ax = axs_j[k] if nchns >1 else axs_j
            ax.set_axis_off()
            vmx = abs(this_filt).max()
            pnl = ax.imshow(this_filt,vmin=-vmx,vmax=vmx)
            cbar = fig.colorbar(pnl, ax=ax)
            cbar.ax.tick_params(labelsize=7)

            if k == 0:
                ax.set_title("Filter: " + str(j), { 'weight' : 'bold' }, fontsize=7)
    
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{get_lay}_" + t_stamp + ".png"
    plt.savefig(pth)

def save_receptive_fields_plot(cfg,device,enc,obs_torch):

    isz = list(obs_torch['obs'].size())[1:]
    outmtx = enc.nl(enc.conv1(obs_torch['obs']))
    osz = list(outmtx.size())[1:]

    nchns = isz[0]
    flts = osz[0]
    rds = 1 + (1 + enc.kernel_size) // 2
    rwsmlt = 2 if flts > 8 else 1 # rows in rf subplot
    fltsdv = flts//rwsmlt

    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv,dpi = 100,figsize = [20,14])

    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,flt,rds,isz)

            for k in range(nchns):

                # Plotting statistics
                rw = k + j*nchns
                ax = axs[rw,i] if flts > 1 else axs[rw] # if one filter - axs is a 1-D array
                ax.set_axis_off()
                vmx = abs(avg[k,:,:]).max()
                pnl = ax.imshow(avg[k,:,:],vmin=-vmx,vmax=vmx)
                fig.colorbar(pnl, ax=ax)

                if k == 0:
                    ax.set_title("Filter: " + str(flt), { 'weight' : 'bold' } )
    
    pth = cfg.train_dir +  "/" + cfg.experiment + "/receptive-fields-" + str(np.datetime64('now')) + ".png"
    plt.savefig(pth)


def spike_triggered_average(dev,enc,flt,rds,isz):

    with torch.no_grad():

        btchsz = [25000] + isz
        cnty = (1+btchsz[2])//2
        cntx = (1+btchsz[3])//2
        mny = cnty - rds
        mxy = cnty + rds
        mnx = cntx - rds
        mxx = cntx + rds
        obsns = torch.randn(size=btchsz,device=dev)
        outmtx = (enc.nl(enc.conv1(obsns)))
        outsz = outmtx.size()
        outs = outmtx[:,flt,outsz[2]//2,outsz[3]//2].cpu()
        obsns1 = obsns[:,:,mny:mxy,mnx:mxx].cpu()
        avg = np.average(obsns1,axis=0,weights=outs)

    return avg
