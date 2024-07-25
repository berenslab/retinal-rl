from typing import Dict, List, Tuple

from omegaconf import DictConfig
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import make_runner
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from torch import Tensor, optim
from torch.utils.data import Dataset

from retinal_rl.models.brain import Brain
from retinal_rl.rl.interface import RLEngine
from retinal_rl.rl.sample_factory.config_defaults import SfDefaults
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from retinal_rl.rl.system.arguments import (add_retinal_env_args,
                                            add_retinal_env_eval_args,
                                            retinal_override_defaults)
from retinal_rl.rl.system.environment import register_retinal_env
from retinal_rl.rl.system.exec import RetinalAlgoObserver


def get_default_cfg(envname: str = "") -> Config: # TODO: get rid of intermediate parser step?!

    mock_argv = ["--env", envname]
    # SF needs an env name in argv.
    # Also, when loading from a checkpoint arguments in argv will not be overridden by arguments defined in the ckpt cfg. 
    parser, cfg = parse_sf_args(mock_argv, evaluation=True)


    add_retinal_env_args(parser) # TODO: Replace with hydra style default to have all in one place & style (sf_config_hydra.yaml?)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)

    return parse_full_cfg(parser, mock_argv)

class SFEngine(RLEngine):
    def train(
        self,
        cfg: DictConfig,
        brain: Brain,
        optimizer: optim.Optimizer,
        train_set: Dataset[Tuple[Tensor, int]],
        test_set: Dataset[Tuple[Tensor, int]],
        completed_epochs: int,
        histories: Dict[str, List[float]],
    ):
        sf_cfg = self.unpack_cfg(
            cfg
        ) 
        # we need to convert to the sample_factory config style since we can not change the function signatures
        # of the library and that uses it _everywhere_

        # Register retinal environments and models.
        register_retinal_env(sf_cfg.env, sf_cfg.input_satiety)
        global_model_factory().register_actor_critic_factory(SampleFactoryBrain)

        # TODO: set currently unused values (if applicable) - else adjust interface!
        # Run simulation
        if not (sf_cfg.dry_run):
            cfg, runner = make_runner(sf_cfg)
            if cfg.online_analysis:
                runner.register_observer(RetinalAlgoObserver(sf_cfg))

            # TODO: set weights of brain through Observer on_init?

            status = runner.init()
            if status == ExperimentStatus.SUCCESS:
                status = runner.run()
            return status
        pass

    def unpack_cfg(self, cfg: DictConfig) -> Tuple[ObsSpace, ActionSpace, Config]:
        inp_shape = (
            *cfg.defaults.dataset.visual_field,
            cfg.defaults.dataset.num_colours,
        )
        obs_space = ObsSpace(shape=inp_shape)  # Get obs space from cfg or brain model
        action_space = ActionSpace()  # TODO: Define action space in cfg
        sf_cfg = get_default_cfg()  # Load Defaults
        # TODO: merge cfg and sf_cfg
        sf_cfg.brain = cfg.brain
        return obs_space, action_space, sf_cfg


def brain_from_actor_critic(actor_critic: SampleFactoryBrain) -> Brain:
    return actor_critic.get_brain()  # TODO: Check if needed
