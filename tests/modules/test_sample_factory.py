import sys

from omegaconf import DictConfig
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.train import make_runner
from sample_factory.utils.attr_dict import AttrDict

sys.path.append(".")
from retinal_rl.rl.sample_factory.environment import register_retinal_env
from retinal_rl.rl.sample_factory.models import SampleFactoryBrain
from runner.frameworks.rl.sf_framework import SFFramework


def test_init_framework(rl_config: DictConfig, data_root: str):
    framework = SFFramework(rl_config, data_root)
    _, runner = make_runner(framework.sf_cfg)
    status = runner.init()
    assert status == ExperimentStatus.SUCCESS


def test_actor_critic_brain(rl_config: DictConfig, data_root: str):
    sf_cfg = SFFramework.to_sf_cfg(rl_config)
    register_retinal_env(sf_cfg.env, data_root, False)
    global_model_factory().register_actor_critic_factory(SampleFactoryBrain)
    env = make_env_func_batched(
        sf_cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0)
    )

    create_actor_critic(sf_cfg, env.observation_space, env.action_space)
