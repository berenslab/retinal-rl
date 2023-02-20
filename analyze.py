import sys

from retinal_rl.system.encoders import register_retinal_model
from retinal_rl.system.environment import register_retinal_envs
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args

from retinal_rl.analysis.processing import get_ac_env,save_onxx,save_simulation,load_simulation
from retinal_rl.analysis.plot import plot_simulation

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

def analyze(cfg):
    """ Final gluing together of all analyses of interest. """
    #print(actor_critic)
    if cfg.simulate:
        ac,env = get_ac_env(cfg)
        save_onxx(cfg,ac,env)
        save_simulation(cfg,ac,env)
        env.close()

    if cfg.plot:
        sim = load_simulation(cfg)
        plot_simulation(sim)


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
