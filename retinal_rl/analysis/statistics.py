import numpy as np
import torch

from sklearn.decomposition import PCA

from openTSNE import TSNE

from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from tqdm import tqdm

from retinal_rl.system.encoders import is_activation

def gaussian_noise_stas(cfg,env,actor_critic,nbtch,nreps,prgrs):
    """
    Returns the receptive fields of every layer of a convnet as computed by spike-triggered averaging.
    """

    enc = actor_critic.encoder.basic_encoder
    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    nclrs,nrws,ncls = list(env.observation_space["obs"].shape)

    obs = env.observation_space.sample()
    nobs = prepare_and_normalize_obs(actor_critic, obs)["obs"]

    btchsz = [nbtch,nclrs,nrws,ncls]

    hrf_size = 1
    hrf_scale = 1
    hrf_shift = 0

    wrf_size = 1
    wrf_scale = 1
    wrf_shift = 0


    stas = {}
    lyridx = 0

    with torch.no_grad():

        for i in tqdm(range(len(enc.conv_head)),desc="Generating STAs",disable=not(prgrs)):

            if is_activation(enc.conv_head[i]):

                subenc = enc.conv_head[0:i+1]

                ochns,hsz,wsz = subenc(nobs).size()

                hidx = hsz//2
                widx = wsz//2

                hmn=hidx*hrf_scale - hrf_shift
                hmx=hmn + hrf_size

                wmn=widx*wrf_scale - wrf_shift
                wmx=wmn + wrf_size

                lyrnm = "layer-" + str(lyridx)
                lyridx += 1

                stas[lyrnm] = np.zeros((ochns,nclrs,hrf_size,wrf_size))

                for j in range(ochns):

                    for _ in range(nreps):

                        obss = torch.randn(size=btchsz,device=dev)
                        obss1 = obss[:,:,hmn:hmx,wmn:wmx].cpu()
                        outs = subenc(obss)[:,j,hidx,widx].cpu()

                        if torch.sum(outs) != 0:
                            stas[lyrnm][j] += np.average(obss1,axis=0,weights=outs)/nreps

            else:

                def double_up(x):
                    if isinstance(x,int): return (x,x)
                    else: return x

                hksz,wksz = double_up(enc.conv_head[i].kernel_size)
                hstrd,wstrd = double_up(enc.conv_head[i].stride)
                hpad,wpad = double_up(enc.conv_head[i].padding)

                hrf_size += (hksz-1)*hrf_scale
                wrf_size += (wksz-1)*wrf_scale

                hrf_shift += hpad*hrf_scale
                wrf_shift += wpad*wrf_scale

                hrf_scale *= hstrd
                wrf_scale *= wstrd

    return stas

def row_zscore(mat):
    return (mat - np.mean(mat,1)[:,np.newaxis])/(np.std(mat,1)[:,np.newaxis]+1e-8)

def fit_tsne_1d(data):
    print('fitting 1d-tSNE...')
    # default openTSNE params
    tsne = TSNE(
        n_components=1,
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    tsne_emb = tsne.fit(data)
    return tsne_emb

def fit_tsne(data):
    print('fitting tSNE...')
    # default openTSNE params
    tsne = TSNE(
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    tsne_emb = tsne.fit(data.T)
    return tsne_emb

def fit_pca(data):
    print('fitting PCA...')
    pca=PCA()
    pca.fit(data)
    embedding = pca.components_.T
    var_exp = pca.explained_variance_ratio_
    return embedding, var_exp

def get_stim_coll(all_health, health_dep=-8, death_dep=30):

    stim_coll = np.diff(all_health)
    stim_coll[stim_coll == health_dep] = 0 # excluding 'hunger' decrease
    stim_coll[stim_coll > death_dep] = 0 # excluding decrease due to death
    stim_coll[stim_coll < -death_dep] = 0
    return stim_coll

#def mei_receptive_fields(cfg,env,actor_critic,nstps=5000,pad=2):
#    """
#    Returns the receptive fields of every layer of a convnet as computed by maximally exciting inputs.
#    """
#
#    enc = actor_critic.encoder.basic_encoder
#    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")
#
#    obs0 = env.observation_space.sample()
#    nobs0 = prepare_and_normalize_obs(actor_critic, obs0)["obs"]
#
#    nclrs,nrws,ncls = list(env.observation_space["obs"].shape)
#    cnty = nrws//2
#    cntx = ncls//2
#
#    nlys = cfg.vvs_depth+2
#    meis = {}
#
#    nrfs = 0
#    for i in range(nlys): nrfs += enc.conv_head[2*i].out_channels
#
#    with tqdm(total=nrfs, desc="Generating MEIs") as pbar:
#
#        for i in range(nlys):
#
#            subenc = enc.conv_head[0:(1+i)*2]
#            ochns,oxsz,oysz = subenc(nobs0).size()
#            rds = pad + ((2*i+1) * enc.kernel_size)//2
#            span = 2*rds
#            mny = cnty - rds
#            mxy = cnty + rds
#            mnx = cntx - rds
#            mxx = cntx + rds
#
#            lyrnm = "layer-" + str(i)
#
#            meis[lyrnm] = np.zeros((ochns,nclrs,span,span))
#
#            for j in range(ochns):
#
#                obs = env.observation_space.sample()
#                nobs = prepare_and_normalize_obs(actor_critic, obs0)["obs"]
#
#                def f(x): return -subenc(x)[j,oxsz//2,oysz//2].cpu()
#
#                nobs.requires_grad_()
#                optimizer = torch.optim.Adam([nobs], lr=0.1)
#
#                for _ in range(nstps):
#
#                    optimizer.zero_grad()
#                    loss = f(nobs)
#                    loss.backward()
#                    optimizer.step()
#                    print(-loss)
#                    if -loss <= 0: break
#                    #list_params.append(params.detach().clone()) #here
#
#                meis[lyrnm][j] = nobs[:,mny:mxy,mnx:mxx].cpu().detach().numpy()
#                pbar.update(1)
#
#        return meis
