import sys
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
import torchscan as ts


from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.utils.attr_dict import AttrDict

from sample_factory.algo.utils.make_env import make_env_func_batched

from retinal_rl.system.brain import register_brain, make_encoder
from retinal_rl.system.environment import register_retinal_env
from retinal_rl.system.arguments import (
    retinal_override_defaults,
    add_retinal_env_args,
    add_retinal_env_eval_args,
)

from retinal_rl.util import (
    fill_in_argv_template,
)

from retinal_rl.system.exec import run_rl


### Main ###


def main():
    """Script entry point."""
    # Register retinal environments and models.

    # Parsing args
    argv = sys.argv[1:]
    # Replace string templates in argv with values from argv.
    argv = fill_in_argv_template(argv)

    # Two-pass building parser and returning cfg : Namespace
    parser, cfg = parse_sf_args(argv, evaluation=True)

    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)

    cfg = parse_full_cfg(parser, argv)

    register_retinal_env(cfg)
    register_brain()

    test_env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode="rgb_array",
    )

    obs_space = test_env.observation_space
    enc = make_encoder(cfg, obs_space).vision_model

    print("Vison Model summary:")
    ts.summary(
        enc, (3, cfg.res_h, cfg.res_w), receptive_field=True
    )  # ,effective_rf_stats=True)
    print("\nEnvironment wrappers:\n")
    # Get string representation of environment wrappers
    print(test_env)

    # Run simulation
    if not (cfg.dry_run):

        status = run_rl(cfg)
        return status


if __name__ == "__main__":
    sys.exit(main())
