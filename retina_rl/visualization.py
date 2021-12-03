# Science
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pygifsicle import optimize

#from torchinfo import summary
#from captum.attr import NeuronGradient


def save_simulation_gif(imgs):

    gif_path = "simulation.gif"

    wrt = imageio.get_writer(gif_path, mode='I',fps=30)

    with wrt as writer:
        for img in imgs:
            writer.append_data(img)

    optimize(gif_path)

def save_receptive_fields_plot(device,enc,isz,osz):

    nchns = isz[0]
    flts = osz[0]
    rds = 3
    rwsmlt = 2
    fltsdv = flts//rwsmlt

    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv)

    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,flt,rds,isz)

            for k in range(nchns):

                # Plotting statistics
                rw = k + j*nchns
                ax = axs[rw,i]
                vmx = abs(avg[k,:,:]).max()
                pnl = ax.imshow(avg[k,:,:],vmin=-vmx,vmax=vmx)
                fig.colorbar(pnl, ax=ax)

                if k == 0:
                    ax.set_title("Filter: " + str(flt), { 'weight' : 'bold' } )

    plt.savefig("receptive-fields.png")


def spike_triggered_average(dev,enc,flt,rds,isz):

    with torch.no_grad():

        btchsz = [20000] + isz
        cnty = btchsz[2]//2
        cntx = btchsz[3]//2
        mny = cnty - rds
        mxy = cnty + rds+1
        mnx = cntx - rds
        mxx = cntx + rds+1
        obsns = torch.randn(size=btchsz,device=dev)
        outmtx = enc.nl2(enc.conv2(enc.nl1(enc.conv1(obsns))))
        outsz = outmtx.size()
        outs = outmtx[:,flt,outsz[2]//2,outsz[3]//2].cpu()
        obsns1 = obsns[:,:,mny:mxy,mnx:mxx].cpu()
        avg = np.average(obsns1,axis=0,weights=outs)

    return avg
