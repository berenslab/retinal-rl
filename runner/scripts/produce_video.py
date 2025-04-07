from pathlib import Path
import sys
from omegaconf import OmegaConf
from runner.frameworks.rl.sf_framework import SFFramework
from sample_factory.enjoy import enjoy

OmegaConf.register_new_resolver("eval", eval)

def create_video(experiment_path: Path):
    # Load the config file
    cfg = OmegaConf.load(experiment_path / "config"/ "config.yaml")
    cfg.path.run_dir = experiment_path

    cfg.logging.use_wandb = False
    cfg.samplefactory.save_video = True
    cfg.samplefactory.no_render = True

    framework = SFFramework(cfg, "cache")
    enjoy(framework.sf_cfg)

experiment_path = Path(sys.argv[1])

if __name__ == '__main__':
    create_video(experiment_path)
