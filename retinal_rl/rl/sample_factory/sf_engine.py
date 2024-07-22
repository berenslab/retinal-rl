from typing import Tuple
from omegaconf import DictConfig
import torch

from retinal_rl.models.brain import Brain
from retinal_rl.rl.interface import RLEngine
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from retinal_rl.rl.sample_factory.config_defaults import SfDefaults
from sample_factory.algo.utils.context import global_model_factory
from retinal_rl.rl.system.environment import register_retinal_env

from sample_factory.train import make_runner
from retinal_rl.rl.system.exec import RetinalAlgoObserver


from sample_factory.algo.utils.misc import ExperimentStatus

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from torch import optim
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List


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
        )  # we need to convert to the sample_factory config style since we can not change the function signatures of the library and that uses it _everywhere_

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

    def unpack_cfg(cfg: DictConfig) -> Tuple[ObsSpace, ActionSpace, Config]:
        inp_shape = (
            *cfg.defaults.dataset.visual_field,
            cfg.defaults.dataset.num_colours,
        )
        obs_space = ObsSpace(shape=inp_shape)  # Get obs space from cfg or brain model
        action_space = ActionSpace()  # TODO: Define action space in cfg
        sf_cfg = SfDefaults()  # Load Defaults
        # TODO: merge cfg and sf_cfg
        sf_cfg.brain = cfg.brain
        return obs_space, action_space, sf_cfg


def brain_from_actor_critic(actor_critic: SampleFactoryBrain) -> Brain:
    return actor_critic.get_brain()  # TODO: Check if needed
