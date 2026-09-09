import logging
import sys

import numpy as np
from omegaconf import DictConfig
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.train import make_runner
from sample_factory.utils.attr_dict import AttrDict
import torch
import torchvision

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
    sf_cfg = AttrDict(sf_cfg.__dict__)  # Convert to AttrDict for easier access
    register_retinal_env(sf_cfg.env, data_root, sf_cfg.get("satiety_input", False), sf_cfg.get("allow_backwards", True))
    global_model_factory().register_actor_critic_factory(SampleFactoryBrain)
    env = make_env_func_batched(
        sf_cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0)
    )

    create_actor_critic(sf_cfg, env.observation_space, env.action_space)

def check_obs_type(obs, dtype):
    if isinstance(obs, dict):
        assert "obs" in obs, "Observation dictionary should contain 'obs' key"
        assert obs["obs"].dtype == dtype, "Observation type mismatch"
    else:
        assert obs.dtype == dtype, "Observation type mismatch"

def test_input_normalization(rl_config: DictConfig, data_root: str):
    sf_cfg = SFFramework.to_sf_cfg(rl_config)
    sf_cfg = AttrDict(sf_cfg.__dict__)  # Convert to AttrDict for easier access
    register_retinal_env(sf_cfg.env, data_root, sf_cfg.get("satiety_input", False), sf_cfg.get("allow_backwards", True))
    global_model_factory().register_actor_critic_factory(SampleFactoryBrain)
    env = make_env_func_batched(
        sf_cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0)
    )

    brain = create_actor_critic(sf_cfg, env.observation_space, env.action_space)
    # Check if the input normalization is applied
    obs,_ = env.reset()
    # assert obs type
    check_obs_type(obs, torch.uint8)
    normalized_obs = brain.normalize_obs(obs)
    check_obs_type(normalized_obs, torch.float32)
    print(normalized_obs['obs'].min(), normalized_obs['obs'].max())

    assert isinstance(normalized_obs, dict)


def test_custom_input_normalization(rl_config: DictConfig, data_root: str):
    sf_cfg = SFFramework.to_sf_cfg(rl_config)
    sf_cfg = AttrDict(sf_cfg.__dict__)  # Convert to AttrDict for easier access
    sf_cfg['transforms'] = [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    sf_cfg['normalize_input'] = False
    register_retinal_env(sf_cfg.env, data_root, sf_cfg.get("satiety_input", False), sf_cfg.get("allow_backwards", True))
    global_model_factory().register_actor_critic_factory(SampleFactoryBrain)
    env = make_env_func_batched(
        sf_cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0)
    )

    brain = create_actor_critic(sf_cfg, env.observation_space, env.action_space)
    # Check if the input normalization is applied
    obs,_ = env.reset()
    # assert obs type
    check_obs_type(obs, torch.uint8)
    normalized_obs = brain.normalize_obs(obs)
    check_obs_type(normalized_obs, torch.float32)

    assert normalized_obs['obs'].min() >= -1.0 and normalized_obs['obs'].max() <= 1.0, "Normalized observation values should be in the range [-1, 1]"

    assert isinstance(normalized_obs, dict)