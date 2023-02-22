import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np

from openTSNE import TSNE

from retinal_rl.analysis.util import analysis_path

def receptive_fields_plot(cfg,env,enc):

    obs = env.reset() # this is the first observation when agent is spawned
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    if cfg.greyscale:
        obs = obs['obs'][:,0,None,:,:] # setting shape of observation to be only single channel
    else:
        obs = obs['obs']

    stas = None
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

    for
    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,lay,flt,rds,isz)

    np.save(analysis_path(cfg,"spike_triggered_averages.npy"), stas, allow_pickle=True)

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

# to plot library
def row_zscore(mat):
    return (mat - np.mean(mat,1)[:,np.newaxis])/(np.std(mat,1)[:,np.newaxis]+1e-8)

## linear decoder analysis
def get_class_accuracy(cfg, ds_out, mode='multi', thr=5, permute=False):
    # mode can be 'multi' or 'bin', thr determines threshold for binarisation, permute will randomly shuffle labels (to get chance preformance)

    X = ds_out['all_fc_act'].T

    if mode == 'multi':
        y = ds_out['all_lab']
    elif mode == 'bin':
        y = ds_out['all_lab']<thr

    perm_str = '' if not permute else 'permuted '

    if permute:
        y = np.random.permutation(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, y_train)

    score_train = logreg.score(X_train, y_train)
    score_test = logreg.score(X_test, y_test)

    return f'{perm_str}{mode} classification scores:\n  -Train: {np.round(score_train,4)}\n  -Test: {np.round(score_test,4)}\n\n'


