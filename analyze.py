import sys
import os

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args

from retinal_rl.analysis.simulation import get_ac_env,generate_simulation,get_checkpoint
from retinal_rl.analysis.statistics import mei_receptive_fields,sta_receptive_fields
from retinal_rl.analysis.util import save_data,load_data,save_onxx,analysis_path,plot_path,data_path
from retinal_rl.analysis.plot import simulation_plot,receptive_field_plots,plot_acts_tsne_stim

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.wandb_utils import init_wandb,finish_wandb
from sample_factory.utils.utils import log

import wandb

def analyze(cfg):

    register_retinal_envs()
    register_retinal_model()

    log.debug("Running analysis: simulate = %s, plot = %s, animate = %s", not(cfg.no_simulate), not(cfg.no_plot), not(cfg.no_animate))

    # Register retinal environments and models.
    checkpoint_dict,cfg = get_checkpoint(cfg)

    if checkpoint_dict is None:
        log.debug("RETINAL RL: No checkpoint found, aborting analysis.")
        sys.exit(1)

    log.debug("RETINAL RL: Checkpoint loaded, preparing environment.")

    ac,env,cfg,envstps = get_ac_env(cfg,checkpoint_dict)

    log.debug("RETINAL RL: Model and environment loaded, preparing simulation.")

    if not os.path.exists(analysis_path(cfg,envstps)):
        os.makedirs(data_path(cfg,envstps))
        os.makedirs(plot_path(cfg,envstps))

    """ Final gluing together of all analyses of interest. """
    if not (cfg.no_simulate):
        save_onxx(cfg,envstps,ac,env)

        stas = sta_receptive_fields(cfg,env,ac,nbtch=1000,nreps=cfg.sta_repeats)
        save_data(cfg,envstps,stas,"stas")

        sim_recs = generate_simulation(cfg,ac,env)
        save_data(cfg,envstps,sim_recs,"sim_recs")

    if not (cfg.no_plot):

        # Load data
        sim_recs = load_data(cfg,envstps,"sim_recs")

        # Single frame of the animation
        fig = plot_acts_tsne_stim(sim_recs)
        pth=plot_path(cfg,envstps,"latent-activations.pdf")

        fig.savefig(pth, bbox_inches="tight")
        if cfg.with_wandb: wandb.log({"latent-activations": wandb.Image(fig)})

        # Single frame of the animation
        fig = simulation_plot(sim_recs,frame_step=cfg.frame_step)
        pth=plot_path(cfg,envstps,"simulation-frame.pdf")

        fig.savefig(pth, bbox_inches="tight")
        if cfg.with_wandb: wandb.log({"simulation-frame": wandb.Image(fig)})

        # STA receptive fields
        stas = load_data(cfg,envstps,"stas")
        figs = receptive_field_plots(stas)

        for ky in figs:
            figs[ky].savefig(plot_path(cfg,envstps,ky + "-sta-receptive-fields.pdf"), bbox_inches="tight")
            if cfg.with_wandb: wandb.log({ky + "-sta-receptive-fields": wandb.Image(figs[ky])})

    if not (cfg.no_animate):

        # Animation
        sim_recs = load_data(cfg,envstps,"sim_recs")
        anim = simulation_plot(sim_recs,animate=True,fps=cfg.fps)
        pth = plot_path(cfg,envstps,"simulation-animation.mp4")

        anim.save(pth, extra_args=["-vcodec", "libx264"] )
        if cfg.with_wandb: wandb.log({"simulation-animation": wandb.Video(pth)})

    env.close()

    return 0


def main():
    """Script entry point."""

    # Two-pass building parser and returning cfg : Namespace
    parser, _ = parse_sf_args(evaluation=True)
    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)
    cfg = parse_full_cfg(parser)

    init_wandb(cfg)

    # Run analysis
    analyze(cfg)

    finish_wandb(cfg)

if __name__ == '__main__':
    sys.exit(main())
