# Science
from functools import cmp_to_key
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pygifsicle import optimize

#from torchinfo import summary
#from captum.attr import NeuronGradient


def save_simulation_gif(cfg,imgs, lay=0):
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir + "/" + cfg.experiment + f"/sim-lay{lay}_" + t_stamp + ".gif"

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
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{lay}_" + t_stamp + ".png"
    plt.savefig(pth)


def spike_triggered_average(dev,enc,lay,flt,rds,isz):

    with torch.no_grad():

        btchsz = [25000] + isz
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
