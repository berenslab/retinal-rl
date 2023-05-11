import numpy as np
import torch

from openTSNE import TSNE

from captum.attr import NeuronGradient

from tqdm import tqdm

from retinal_rl.util import encoder_out_size,rf_size_and_start

def gaussian_noise_stas(cfg,env,actor_critic,nbtch,nreps,prgrs):
    """
    Returns the receptive fields of every layer of a convnet as computed by spike-triggered averaging.
    """

    enc = actor_critic.encoder.basic_encoder
    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    nclrs,hght,wdth = list(env.observation_space["obs"].shape)
    ochns = nclrs

    btchsz = [nbtch,nclrs,hght,wdth]

    stas = {}

    repttl = len(enc.conv_head) * nreps
    mdls = []

    with torch.no_grad():

        with tqdm(total=repttl,desc="Generating STAs",disable=not(prgrs)) as pbar:

            for lyrnm, mdl in enc.conv_head.named_children():

                mdls.append(mdl)
                subenc = torch.nn.Sequential(*mdls)

                # check if mdl has out channels
                if hasattr(mdl,'out_channels'):
                    ochns = mdl.out_channels
                hsz,wsz = encoder_out_size(subenc,hght,wdth)

                hidx = (hsz-1)//2
                widx = (wsz-1)//2

                hrf_size,wrf_size,hmn,wmn = rf_size_and_start(subenc,hidx,widx)

                hmx=hmn + hrf_size
                wmx=wmn + wrf_size

                stas[lyrnm] = np.zeros((ochns,nclrs,hrf_size,wrf_size))

                for _ in range(nreps):

                    pbar.update(1)

                    for j in range(ochns):

                        obss = torch.randn(size=btchsz,device=dev)
                        obss1 = obss[:,:,hmn:hmx,wmn:wmx].cpu()
                        outs = subenc(obss)[:,j,hidx,widx].cpu()

                        if torch.sum(outs) != 0:
                            stas[lyrnm][j] += np.average(obss1,axis=0,weights=outs)/nreps

    return stas

def gradient_receptive_fields(cfg,env,actor_critic,prgrs):
    """
    Returns the receptive fields of every layer of a convnet as computed by neural gradients.
    """

    enc = actor_critic.encoder.basic_encoder
    dev = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    nclrs,hght,wdth = list(env.observation_space["obs"].shape)
    ochns = nclrs

    imgsz = [1,nclrs,hght,wdth]
    # Obs requires grad
    obs = torch.zeros(size=imgsz,device=dev,requires_grad=True)

    stas = {}

    repttl = len(enc.conv_head)
    mdls = []

    with torch.no_grad():

        with tqdm(total=repttl,desc="Generating Attributions",disable=not(prgrs)) as pbar:

            for lyrnm, mdl in enc.conv_head.named_children():

                gradient_calculator = NeuronGradient(enc,mdl)
                mdls.append(mdl)
                subenc = torch.nn.Sequential(*mdls)

                # check if mdl has out channels
                if hasattr(mdl,'out_channels'):
                    ochns = mdl.out_channels
                hsz,wsz = encoder_out_size(subenc,hght,wdth)

                hidx = (hsz-1)//2
                widx = (wsz-1)//2

                hrf_size,wrf_size,hmn,wmn = rf_size_and_start(subenc,hidx,widx)

                hmx=hmn + hrf_size
                wmx=wmn + wrf_size

                stas[lyrnm] = np.zeros((ochns,nclrs,hrf_size,wrf_size))

                pbar.update(1)

                for j in range(ochns):

                    grad = gradient_calculator.attribute(obs,(j,hidx,widx))[0,:,hmn:hmx,wmn:wmx].cpu().numpy()

                    stas[lyrnm][j] = grad

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


def get_stim_coll(all_health, health_dep=-8, death_dep=30):

    stim_coll = np.diff(all_health)
    stim_coll[stim_coll == health_dep] = 0 # excluding 'hunger' decrease
    stim_coll[stim_coll > death_dep] = 0 # excluding decrease due to death
    stim_coll[stim_coll < -death_dep] = 0
    return stim_coll

#def fit_pca(data):
#    print('fitting PCA...')
#    pca=PCA()
#    pca.fit(data)
#    embedding = pca.components_.T
#    var_exp = pca.explained_variance_ratio_
#    return embedding, var_exp
#
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
