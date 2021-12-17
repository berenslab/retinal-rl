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

def save_receptive_fields_plot(cfg,device,enc,obs_torch):

    isz = list(obs_torch['obs'].size())[1:]
    outmtx = enc.nl(enc.conv1(obs_torch['obs']))
    osz = list(outmtx.size())[1:]

    nchns = isz[0]
    flts = osz[0]
    rds = 1 + (1 + enc.kernel_size) // 2
    rwsmlt = 2
    fltsdv = flts//rwsmlt

    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv,dpi = 100,figsize = [20,14])

    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,flt,rds,isz)

            for k in range(nchns):

                # Plotting statistics
                rw = k + j*nchns
                ax = axs[rw,i]
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
