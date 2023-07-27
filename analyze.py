import sys
import os

import matplotlib.pyplot as plt

from retinal_rl.system.brain import register_brain
from retinal_rl.system.environment import register_retinal_env
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args

from retinal_rl.analysis.simulation import get_brain_env,generate_simulation,get_checkpoint
from retinal_rl.analysis.statistics import gaussian_noise_stas,gradient_receptive_fields
from retinal_rl.util import save_data,load_data,save_onnx,analysis_path,plot_path,data_path
from retinal_rl.analysis.plot import simulation_plot,receptive_field_plots

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.utils import log

def analyze(cfg,progress_bar=True):

    register_retinal_env(cfg)
    register_brain()

    log.debug("Running analysis: simulate = %s, plot = %s, animate = %s", cfg.simulate, cfg.plot, cfg.animate)

    # Register retinal environments and models.
    checkpoint_dict,cfg = get_checkpoint(cfg)

    cfg.device = "cpu"

    if checkpoint_dict is None:
        log.debug("RETINAL RL: No checkpoint found, aborting analysis.")
        sys.exit(1)

    log.debug("RETINAL RL: Checkpoint loaded, preparing environment.")

    brain,env,cfg,envstps = get_brain_env(cfg,checkpoint_dict)

    if cfg.analysis_name is None:
        ana_name = "env_steps-" + str(envstps)
    else:
        ana_name = cfg.analysis_name

    rf_algs = ["grads"]
    log.debug("RETINAL RL: Model and environment loaded, preparing simulation.")

    if not os.path.exists(analysis_path(cfg,ana_name)):
        os.makedirs(data_path(cfg,ana_name))

    """ Final gluing together of all analyses of interest. """
    if cfg.simulate:

        log.debug("RETINAL RL: Running analysis simulations.")

        inpts,sim_recs = generate_simulation(cfg,brain,env,prgrs=progress_bar)

        save_onnx(cfg,ana_name,brain,inpts)

        save_data(cfg,ana_name,sim_recs,"sim_recs")

    if cfg.receptive_fields:

        log.debug("RETINAL RL: Analyzing receptive fields.")

        for alg in rf_algs:

            if not os.path.exists(os.path.join(plot_path(cfg,ana_name),alg+"_rfs")):
                os.makedirs(os.path.join(plot_path(cfg,ana_name),alg+"_rfs"))

            if alg == "stas":
                stas = gaussian_noise_stas(cfg,env,brain,nbtch=200,nreps=cfg.sta_repeats,prgrs=progress_bar)
                save_data(cfg,ana_name,stas,"stas")

            if alg == "grads":
                grads = gradient_receptive_fields(cfg,env,brain,prgrs=progress_bar)
                save_data(cfg,ana_name,grads,"grads")

    if cfg.plot:

        log.debug("RETINAL RL: Plotting analysis simulations.")

        # Load data
        sim_recs = load_data(cfg,ana_name,"sim_recs")

        # Single frame of the animation
        # fig = plot_acts_tsne_stim(sim_recs)
        # pth=plot_path(cfg,ana_name,"latent-activations.png")

        # fig.savefig(pth, bbox_inches="tight")
        # plt.close()

        # Single frame of the animation
        fig = simulation_plot(sim_recs,frame_step=cfg.frame_step,prgrs=progress_bar)
        pth=plot_path(cfg,ana_name,"simulation-frame.png")

        fig.savefig(pth, bbox_inches="tight")
        plt.close()

        for alg in rf_algs:

            rfs = load_data(cfg,ana_name,alg)

            for ky in rfs.keys():

                lyr = rfs[ky]
                fig = receptive_field_plots(lyr)
                fig.savefig(plot_path(cfg,ana_name,alg+"_rfs/" + alg + "-" + ky + ".png"), bbox_inches="tight")
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

    # Register retinal environments and models.

    # Parsing args
    argv = sys.argv[1:]

    # Two-pass building parser and returning cfg : Namespace
    parser,cfg = parse_sf_args(argv,evaluation=True)

    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)

    cfg = parse_full_cfg(parser, argv)

    # Run analysis
    analyze(cfg)

if __name__ == '__main__':
    sys.exit(main())
