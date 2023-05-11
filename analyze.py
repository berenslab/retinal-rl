import sys
import os

import matplotlib.pyplot as plt

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args

from retinal_rl.analysis.simulation import get_ac_env,generate_simulation,get_checkpoint
from retinal_rl.analysis.statistics import gaussian_noise_stas,gradient_receptive_fields
from retinal_rl.util import save_data,load_data,save_onxx,analysis_path,plot_path,data_path
from retinal_rl.analysis.plot import simulation_plot,receptive_field_plots,plot_acts_tsne_stim

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.utils import log

def analyze(cfg,progress_bar=True):

    register_retinal_envs()
    register_retinal_model()

    log.debug("Running analysis: simulate = %s, plot = %s, animate = %s", cfg.simulate, cfg.plot, cfg.animate)

    # Register retinal environments and models.
    checkpoint_dict,cfg = get_checkpoint(cfg)

    if checkpoint_dict is None:
        log.debug("RETINAL RL: No checkpoint found, aborting analysis.")
        sys.exit(1)

    log.debug("RETINAL RL: Checkpoint loaded, preparing environment.")

    ac,env,cfg,envstps = get_ac_env(cfg,checkpoint_dict)

    if cfg.analysis_name is None:
        ana_name = "env_steps-" + str(envstps)
    else:
        ana_name = cfg.analysis_name

    log.debug("RETINAL RL: Model and environment loaded, preparing simulation.")

    if not os.path.exists(analysis_path(cfg,ana_name)):
        os.makedirs(data_path(cfg,ana_name))
        os.makedirs(os.path.join(plot_path(cfg,ana_name),"sta_rfs"))
        os.makedirs(os.path.join(plot_path(cfg,ana_name),"grad_rfs"))

    save_onxx(cfg,ana_name,ac,env)

    """ Final gluing together of all analyses of interest. """
    if cfg.simulate:

        log.debug("RETINAL RL: Running analysis simulations.")

        sim_recs = generate_simulation(cfg,ac,env,prgrs=progress_bar)
        save_data(cfg,ana_name,sim_recs,"sim_recs")

    if cfg.receptive_fields:

        log.debug("RETINAL RL: Analyzing receptive fields.")

        stas = gaussian_noise_stas(cfg,env,ac,nbtch=200,nreps=cfg.sta_repeats,prgrs=progress_bar)
        save_data(cfg,ana_name,stas,"stas")

        grads = gradient_receptive_fields(cfg,env,ac,prgrs=progress_bar)
        save_data(cfg,ana_name,grads,"grads")

    if cfg.plot:

        log.debug("RETINAL RL: Plotting analysis simulations.")

        # Load data
        sim_recs = load_data(cfg,ana_name,"sim_recs")

        # Single frame of the animation
        fig = plot_acts_tsne_stim(sim_recs)
        pth=plot_path(cfg,ana_name,"latent-activations.png")

        fig.savefig(pth, bbox_inches="tight")
        plt.close()

        # Single frame of the animation
        fig = simulation_plot(sim_recs,frame_step=cfg.frame_step,prgrs=progress_bar)
        pth=plot_path(cfg,ana_name,"simulation-frame.png")

        fig.savefig(pth, bbox_inches="tight")
        plt.close()

        # STA receptive fields
        stas = load_data(cfg,ana_name,"stas")

        for ky in stas:

            lyr = stas[ky]
            fig = receptive_field_plots(lyr)
            fig.savefig(plot_path(cfg,ana_name,"sta_rfs/" + ky + ".png"), bbox_inches="tight")
            plt.close()

        grads = load_data(cfg,ana_name,"grads")

        for ky in grads:

            lyr = grads[ky]
            fig = receptive_field_plots(lyr)
            fig.savefig(plot_path(cfg,ana_name,"grad_rfs/" + ky + ".png"), bbox_inches="tight")
            plt.close()


    if cfg.animate:

        log.debug("RETINAL RL: Animating analysis simulations.")

        # Animation
        sim_recs = load_data(cfg,ana_name,"sim_recs")
        anim = simulation_plot(sim_recs,animate=True,fps=cfg.fps,prgrs=progress_bar)
        pth = plot_path(cfg,ana_name,"simulation-animation.mp4")

        anim.save(pth, extra_args=["-vcodec", "libx264"] )

    env.close()

    return envstps


def main():
    """Script entry point."""

    # Two-pass building parser and returning cfg : Namespace
    parser, _ = parse_sf_args(evaluation=True)
    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)
    cfg = parse_full_cfg(parser)

    # Run analysis
    analyze(cfg)

if __name__ == '__main__':
    sys.exit(main())
