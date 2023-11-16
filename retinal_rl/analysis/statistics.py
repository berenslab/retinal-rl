import torch.nn as nn
import torch

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from openTSNE import TSNE

from captum.attr import NeuronGradient

from tqdm import tqdm

from retinal_rl.util import encoder_out_size,rf_size_and_start
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

def gaussian_noise_stas(cfg,env,actor_critic,nbtch,nreps,prgrs):
    """
    Returns the receptive fields of every layer of a convnet as computed by spike-triggered averaging.
    """

    enc = actor_critic.encoder.vision_model
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

    enc = actor_critic.encoder.vision_model
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
        perplexity=20,
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

class RGClassifier(nn.Module):
    def __init__(self, brain, num_classes):
        super(RGClassifier, self).__init__()
        # conv_head up to rgc_averages
        all_layers = list(brain.valnet.encoder.vision_model.conv_head.children())
        first_six_layers = all_layers[:6]
        self.encoder = nn.Sequential(*first_six_layers)
        self.brain = brain
        self.softmax = nn.Softmax(dim=1)
        # Compute the output size of the encoder
        
        cout_hght,cout_wdth = encoder_out_size(self.encoder, 120, 160)
        self.conv_head_out_size = cout_hght*cout_wdth*self.brain.valnet.encoder.vision_model.rgc_chans

        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze the encoder parameters

        self.classifier = nn.Linear(self.conv_head_out_size, num_classes)

    def forward(self, obs):
        x = self.encoder(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        weights = self.classifier(x)
        prbs = self.softmax(weights)
        return prbs

class V1Classifier(nn.Module):
    def __init__(self, brain, num_classes):
        super(V1Classifier, self).__init__()
        self.encoder = brain.valnet.encoder.vision_model.conv_head
        self.brain = brain
        self.softmax = nn.Softmax(dim=1)
        self.conv_head_out_size = brain.valnet.encoder.vision_model.conv_head_out_size
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze the encoder parameters

        self.classifier = nn.Linear(self.conv_head_out_size, num_classes)

    def forward(self, obs):
        x = self.encoder(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        weights = self.classifier(x)
        prbs = self.softmax(weights)
        return prbs

class VisionClassifier(nn.Module):
    def __init__(self, brain, num_classes):
        super(VisionClassifier, self).__init__()
        self.encoder = brain.valnet.encoder.vision_model
        self.brain = brain
        self.softmax = nn.Softmax(dim=1)
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze the encoder parameters

        self.classifier = nn.Linear(brain.valnet.encoder.encoder_out_size, num_classes)

    def forward(self, obs):
        x = self.encoder(obs)
        return self.classifier(x)

class BrainClassifier(nn.Module):
    def __init__(self, brain, num_classes):
        super(BrainClassifier, self).__init__()
        self.encoder = brain.valnet.encoder
        self.core = brain.valnet.core
        self.fake_rnn_states = brain.valnet.fake_rnn_states
        self.brain = brain
        self.softmax = nn.Softmax(dim=1)
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze the encoder parameters

        self.classifier = nn.Linear(brain.valnet.encoder.encoder_out_size, num_classes)

    def forward(self, obs):
        obs_dict = {"obs": obs}
        nobs_dict = obs_dict
        x = self.encoder(nobs_dict)
        features, _ = self.core(x,self.fake_rnn_states)
        weights = self.classifier(features)
        return self.softmax(weights)

class LinearClassifier(nn.Module):

    def __init__(self, num_classes):
        nn.Module.__init__(self)
        self.classifier = nn.Linear(3*120*160, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs):
        # Flatten the input
        features = obs.view(obs.shape[0], -1)
        return self.classifier(features)


def evaluate_brain(cfg,model, epochs=50, lr=3e-4, batch_size=64):
    if cfg.classification == "mnist":
        train_path='resources/classification/mnist_train_overlay'
        test_path='resources/classification/mnist_test_overlay'
    elif cfg.classification == "cifar10":
        train_path='resources/classification/cifar_train_overlay'
        test_path='resources/classification/cifar_test_overlay'
    else:
        raise NotImplementedError("Only mnist and cifar are supported")
    # Data loading
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = ImageFolder(train_path, transform=transform)
    test_data = ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training function
    def train_model(model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    # Evaluation function
    def evaluate_model(model, test_loader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / len(test_data)
        return total_loss / len(test_loader), accuracy

    # Training loop
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

    # Evaluation
    test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return test_loss, accuracy

# Example usage:
# evaluate_brain(brain)

