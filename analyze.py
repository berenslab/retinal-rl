# System
import sys

# Numerics
import torch
import numpy as np

# retinal-rl
from retinal_rl.visualization import *
from retinal_rl.activations import *


def analyze(cfg):

    cfg = load_from_checkpoint(cfg)
    max_num_frames = cfg.analyze_max_num_frames
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda') # this line should go above the get_acts_actor function
    
    # getting encoder, environment info and activations of layers during experience
    enc, env_infos, all_acts_dict_enc = get_acts_environment(cfg, max_num_frames=max_num_frames) # all_acts_dict (keys correspond to layers)

    # always make simulation gif
    save_simulation_gif(cfg,all_acts_dict_enc['imgs'][:1000]) # only first 1000 frames due to .gif resource limitations
    
    # always plot receptive fields
    n_conv_lay = len(enc.conv_head)//2
    for lay in range(1, n_conv_lay+1):
        save_receptive_fields_plot(cfg,device,enc,lay,all_acts_dict_enc['obs_torch'])

    # optional analysis (activations in environment and to dataset images)
    if cfg.analyze_acts == 'environment':        
        # visualizing conv layers
        for lay in range(n_conv_lay):
            save_activations_gif(cfg, all_acts_dict_enc['imgs'][:1000], all_acts_dict_enc['conv_acts'][lay][:1000], lay) # only first 1000 frames due to .gif resource limitations
        # visualizing PCA of fully connected layer
        plot_PCA_env(cfg, all_acts_dict_enc['imgs'], env_infos, all_acts_dict_enc['fc_acts'], all_acts_dict_enc['v_acts'], n_pcs=50)
        # visualizing tSNE of fully connected layer and rnn
        plot_tsne_env(cfg, all_acts_dict_enc)

    if cfg.analyze_acts == 'dataset':
        
        if cfg.analyze_ds_name == 'CIFAR':
            rewards_dict = {0:2, 1:1, 2:4, 3:5, 4:7, 5:6, 6:8, 7:9, 8:3, 9:0} # defined by the class-reward assignments in the .wad file
        elif cfg.analyze_ds_name == 'MNIST':
            rewards_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9} # defined by the class-reward assignments in the .wad file (matching digits in case of mnist)

        all_acts_dict_ds = get_acts_dataset(cfg, enc, cfg.analyze_ds_name, rewards_dict)
        plot_dimred_ds(cfg, all_acts_dict_ds['fc_acts_np'], all_acts_dict_ds['all_lab_np'])
        
        # TODO: add function for classifier analysis
    
    elif cfg.analyze_acts != 'False':  # if anything other than 'False', 'environment', 'mnist' or 'cifar'
        print('\n\nInvalid cfg.analyze_acts input, terminating script.\n\n')


def main():
    """Script entry point."""
    register_retinal_environment()
    register_encoders()
    cfg = custom_parse_args(evaluation=True)
    analyze(cfg)


if __name__ == '__main__':
    sys.exit(main())
