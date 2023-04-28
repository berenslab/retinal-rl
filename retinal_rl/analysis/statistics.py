import numpy as np
import torch

from sklearn.decomposition import PCA

from openTSNE import TSNE

from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from torch_receptive_field import receptive_field
from retinal_rl.system.encoders import is_activation

from tqdm.auto import tqdm

def sta_receptive_fields(cfg,env,actor_critic,nbtch,nreps):
    """
    Returns the receptive fields of every layer of a convnet as computed by spike-triggered averaging.
    """

    enc = actor_critic.encoder.basic_encoder
    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    nclrs,nrws,ncls = list(env.observation_space["obs"].shape)
    cnty = nrws//2
    cntx = ncls//2

    obs = env.observation_space.sample()
    nobs = prepare_and_normalize_obs(actor_critic, obs)["obs"]

    btchsz = [nbtch,nclrs,nrws,ncls]
    rfsz = 0

    stas = {}
    lyridx = 0

    with torch.no_grad():

        for i in range(len(enc.conv_head)):

            if is_activation(enc.conv_head[i]):

                subenc = enc.conv_head[0:i+1]

                ochns,oxsz,oysz = subenc(nobs).size()
                rds = rfsz//2
                mny = cnty - rds
                mxy = mny + rfsz
                mnx = cntx - rds
                mxx = mnx + rfsz

                lyrnm = "layer-" + str(lyridx)
                lyridx += 1

                stas[lyrnm] = np.zeros((ochns,nclrs,rfsz,rfsz))

                for j in range(ochns):

                    for _ in range(nreps):

                        obss = torch.randn(size=btchsz,device=dev)
                        obss1 = obss[:,:,mny:mxy,mnx:mxx].cpu()
                        outs = subenc(obss)[:,j,oxsz//2,oysz//2].cpu()

                        if torch.sum(outs) != 0:
                            stas[lyrnm][j] += np.average(obss1,axis=0,weights=outs)/nreps

            elif i==0:

                rfsz += enc.conv_head[0].kernel_size[0]

            else:

                ksz = enc.conv_head[i].kernel_size
                if not(isinstance(ksz,int)): ksz = ksz[0]
                rfsz += ksz-1


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

def mei_receptive_fields(cfg,env,actor_critic,nstps=5000,pad=2):
    """
    Returns the receptive fields of every layer of a convnet as computed by maximally exciting inputs.
    """

    enc = actor_critic.encoder.basic_encoder
    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    obs0 = env.observation_space.sample()
    nobs0 = prepare_and_normalize_obs(actor_critic, obs0)["obs"]

    nclrs,nrws,ncls = list(env.observation_space["obs"].shape)
    cnty = nrws//2
    cntx = ncls//2

    nlys = cfg.vvs_depth+2
    meis = {}

    nrfs = 0
    for i in range(nlys): nrfs += enc.conv_head[2*i].out_channels

    with tqdm(total=nrfs, desc="Generating MEIs") as pbar:

        for i in range(nlys):

            subenc = enc.conv_head[0:(1+i)*2]
            ochns,oxsz,oysz = subenc(nobs0).size()
            rds = pad + ((2*i+1) * enc.kernel_size)//2
            span = 2*rds
            mny = cnty - rds
            mxy = cnty + rds
            mnx = cntx - rds
            mxx = cntx + rds

            lyrnm = "layer-" + str(i)

            meis[lyrnm] = np.zeros((ochns,nclrs,span,span))

            for j in range(ochns):

                obs = env.observation_space.sample()
                nobs = prepare_and_normalize_obs(actor_critic, obs0)["obs"]

                def f(x): return -subenc(x)[j,oxsz//2,oysz//2].cpu()

                nobs.requires_grad_()
                optimizer = torch.optim.Adam([nobs], lr=0.1)

                for _ in range(nstps):

                    optimizer.zero_grad()
                    loss = f(nobs)
                    loss.backward()
                    optimizer.step()
                    print(-loss)
                    if -loss <= 0: break
                    #list_params.append(params.detach().clone()) #here

                meis[lyrnm][j] = nobs[:,mny:mxy,mnx:mxx].cpu().detach().numpy()
                pbar.update(1)

        return meis
