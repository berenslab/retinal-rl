import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
from retinal_rl.models.cognition.classification import FullyConnected
from retinal_rl.models.vision.retinal_model import RetinalModel
from retinal_rl.models.brain import Brain

@hydra.main(config_path=".", config_name="config")
def initialize_brain(cfg: DictConfig):
    # Initialize the Brain model
    print(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    brain = Brain(**cfg_dict)
    
    # Print the configuration for verification
    
    # Create a directory to save the initialized model
    save_dir = os.path.join(os.getcwd(), "initialized_brain")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the brain model
    brain.save(save_dir)
    print(f"Brain model saved to {save_dir}")

if __name__ == "__main__":
    initialize_brain()

