import sys

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.train import make_runner
from sample_factory.utils.attr_dict import AttrDict
import torch

sys.path.append(".")
from retinal_rl.rl.loss import build_context
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

def test_env_transforms(rl_config: DictConfig, data_root: str):
    sf_cfg = SFFramework.to_sf_cfg(rl_config)
    source_transforms = torch.nn.Sequential(
        *instantiate(rl_config.dataset.source_transforms)
    )
    noise_transforms = torch.nn.Sequential(
        *instantiate(rl_config.dataset.noise_transforms)
    )
    register_retinal_env(
        sf_cfg.env,
        data_root,
        sf_cfg.input_satiety,
        sf_cfg.allow_backwards,
        source_transforms=source_transforms,
        noise_transforms=noise_transforms,
    )
    env = make_env_func_batched(
        sf_cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0)
    )
    vision_shape = env.observation_space["obs"].shape
    noise_vs_source = []

    def check_obs(obs):
        assert "obs_pre_source" in obs
        assert "obs_pre_noise" in obs
        assert obs["obs"].shape[1:] == vision_shape
        assert obs["obs_pre_source"].shape[1:] == vision_shape
        assert obs["obs_pre_noise"].shape[1:] == vision_shape
        # the source transform (Normalize) always changes both dtype and values
        assert not torch.equal(
            obs["obs_pre_source"].float(), obs["obs_pre_noise"].float()
        )
        noise_vs_source.append(torch.equal(obs["obs_pre_noise"], obs["obs"]))

    obs, _ = env.reset()
    check_obs(obs)

    action = torch.tensor([[space.sample() for space in env.action_space.spaces]])
    for _ in range(9):
        obs, _, _, _, _ = env.step(action)
        check_obs(obs)

    # The noise transform (Blur) samples a continuous factor in (0, 2] and is a
    # no-op only on spatially-uniform frames (e.g. a black loading frame), so
    # across enough steps it must visibly change at least one real frame.
    assert not all(noise_vs_source)


def test_env_transforms_during_training(
    rl_config: DictConfig, data_root: str, monkeypatch
):
    contexts = []

    def spy_build_context(*args, **kwargs):
        context = build_context(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(
        "retinal_rl.rl.sample_factory.learner.build_context", spy_build_context
    )

    framework = SFFramework(rl_config, data_root)
    sf_cfg = framework.sf_cfg

    # Shrink the run so it completes in a few seconds. serial_mode is required
    # (not just for speed): the learner otherwise runs in a separate process,
    # where the monkeypatch above would not apply.
    sf_cfg.serial_mode = True
    sf_cfg.async_rl = False
    sf_cfg.num_workers = 1
    sf_cfg.num_envs_per_worker = 2
    sf_cfg.worker_num_splits = 1
    sf_cfg.decorrelate_experience_max_seconds = 0
    sf_cfg.decorrelate_envs_on_one_worker = False
    sf_cfg.rollout = 8
    sf_cfg.recurrence = 8
    sf_cfg.batch_size = 16
    sf_cfg.train_for_env_steps = 64
    sf_cfg.save_every_sec = 1000

    _, runner = make_runner(sf_cfg)
    status = runner.init()
    assert status == ExperimentStatus.SUCCESS

    status = runner.run()
    assert status == ExperimentStatus.SUCCESS

    # build_context must actually have run, and with the noise branch taken:
    # i.e. the real buffered minibatch carried "obs_pre_noise" all the way
    # through sample-factory's rollout buffers, not just at the raw env level.
    assert len(contexts) > 0
    for context in contexts:
        assert context.inputs is not None
        assert context.sources.shape == context.inputs.shape
        assert not torch.equal(context.sources, context.inputs)
