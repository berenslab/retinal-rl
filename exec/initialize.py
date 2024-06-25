import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from retinal_rl.models.brain import BrainConfig,Brain

@hydra.main(config_path="../resources/config", config_name="brain", version_base=None)
def initialize(cfg: DictConfig):

    instantiated_circuits = {}
    print(cfg.circuits)
    for crcnm, crcfg in cfg.circuits.items():
        instantiated_circuits[crcnm] = instantiate(crcfg)

    brncfg = BrainConfig(
                name=cfg.name,
                circuits=instantiated_circuits,
                sensors=cfg.sensors,
                connections=cfg.connections
                    )

    brain = Brain(brncfg)

    # Run the scan
    brain.scan_circuits()

    # # Create a directory to save the model and config
    # save_dir = os.path.join(os.getcwd(), "saved_brain")
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # Save the brain model
    # brain.save(save_dir)
    #
    # # Save the merged configuration
    # config_save_path = os.path.join(save_dir, "merged_config.yaml")
    # with open(config_save_path, "w") as f:
    #     f.write(OmegaConf.to_yaml(cfg))
    #
    # print(f"Brain model and configuration saved to {save_dir}")

if __name__ == "__main__":
    initialize()

