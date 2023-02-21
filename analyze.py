import sys
import os

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args

from retinal_rl.analysis.simulation import get_ac_env,save_simulation
from retinal_rl.analysis.util import load_simulation,save_onxx,analysis_path
from retinal_rl.analysis.plot import simulation_plot

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

def analyze(cfg):
    """ Final gluing together of all analyses of interest. """

    if not os.path.exists(analysis_path(cfg)):
        os.makedirs(analysis_path(cfg))

    if cfg.simulate:
        ac,env = get_ac_env(cfg)
        save_onxx(cfg,ac,env)
        save_simulation(cfg,ac,env)
        env.close()

    if cfg.plot:
        sim = load_simulation(cfg)
        fig = simulation_plot(sim)
        fig.savefig(analysis_path(cfg,"analysis-frame.pdf"), bbox_inches="tight")

    if cfg.animate:
        sim = load_simulation(cfg)
        anim = simulation_plot(sim,animate=True,fps=cfg.fps)
        anim.save(analysis_path(cfg,"simulation-animation.mp4"), extra_args=["-vcodec", "libx264"] )



def main():
    """Script entry point."""
    # Register retinal environments and models.
    register_retinal_envs()
    register_retinal_model()

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
