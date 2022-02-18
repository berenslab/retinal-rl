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
    enc, env_infos, all_acts_dict = get_acts_environment(cfg, max_num_frames=max_num_frames) # all_acts_dict (keys correspond to layers)
    # TODO: write a similar function here but to get the activations for 'dataset'

    # always make simulation gif
    save_simulation_gif(cfg,all_acts_dict['imgs'][:1000]) # only first 1000 frames due to .gif resource limitations
    
    # always plot receptive fields
    n_conv_lay = len(enc.conv_head)//2
    for lay in range(1, n_conv_lay+1):
        save_receptive_fields_plot(cfg,device,enc,lay,all_acts_dict['obs_torch'])

    # optional analysis (activations in environment and to dataset images)
    if cfg.analyze_acts == 'environment':        
        # visualizing conv layers
        for lay in range(n_conv_lay):
            save_activations_gif(cfg, all_acts_dict['imgs'][:1000], all_acts_dict['conv_acts'][lay][:1000], lay) # only first 1000 frames due to .gif resource limitations
        # visualizing PCA of fully connected layer
        plot_PCA(cfg, all_acts_dict['imgs'], env_infos, all_acts_dict['fc_acts'], all_acts_dict['v_acts'], n_pcs=50)
        # visualizing tSNE of fully connected layer and rnn
        plot_tsne(cfg, all_acts_dict)

    # TODO: add 'dataset' input compatibility for analyzing layer activations
    # elif cfg.analyze_acts == 'mnist' or cfg.analyze_acts == 'cifar':
    #     get_acts_dataset(enc, cfg.analyze_acts) # define function
    
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
