from dataclasses import dataclass
from pathlib import Path

import torch

from retinal_rl.analysis import default as default_ana
from retinal_rl.analysis import receptive_fields
from retinal_rl.analysis.plot import FigureLogger
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import ContextT, Objective

### Infrastructure ###


@dataclass
class AnalysesCfg: # TODO: Unify analyze scripts & cfg -> common 'base_analysis'? How much is actually framework specific?
    run_dir: Path
    plot_dir: Path
    checkpoint_plot_dir: Path
    data_dir: Path
    use_wandb: bool

    def __post_init__(self):
        self.analyses_dir = Path(self.data_dir) / "analyses"

        # Ensure all dirs exist
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_plot_dir.mkdir(parents=True, exist_ok=True)
        self.analyses_dir.mkdir(parents=True, exist_ok=True)


### Analysis ###


def analyze(
    cfg: AnalysesCfg,
    device: torch.device,
    brain: Brain,
    objective: Objective[ContextT],
    histories: dict[str, list[float]],
    epoch: int,
    copy_checkpoint: bool = False,
):
    log = FigureLogger(
        cfg.use_wandb, cfg.plot_dir, cfg.checkpoint_plot_dir, cfg.run_dir
    )

    log.plot_and_save_histories(histories)

    # perform different analyses, plot and log them
    input_shape, rf_result = receptive_fields.analyze(brain, device)
    receptive_fields.plot(
        log,
        rf_result,
        epoch,
        copy_checkpoint,
    )
    log.save_dict(cfg.analyses_dir / f"receptive_fields_epoch_{epoch}.json", rf_result)

    if epoch == 0:
        default_ana.initialization_plots(log, brain, objective, input_shape, rf_result)

    log.plot_and_save_histories(histories, save_always=True)
