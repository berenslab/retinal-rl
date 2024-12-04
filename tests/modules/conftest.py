import os
import shutil
import sys
import time
from typing import Generator

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf

sys.path.append(".")
from runner.util import search_conf

OmegaConf.register_new_resolver("eval", eval)


# TODO: make this independent of whether templates are in the right place or not etc
def config(experiment: str) -> DictConfig:
    with hydra.initialize(config_path="../../config/base", version_base=None):
        config = hydra.compose(
            "config", overrides=[f"+experiment={experiment}", "system.device=cpu"]
        )

        # replace the paths that are normally set via HydraConfig
        config.path.run_dir = f"tmp{hash(time.time())}"
        config.sweep.command[-2] = experiment

        # check whether there's still values to be interpolated through hydra
        hydra_values = search_conf(
            OmegaConf.to_container(config, resolve=False), "hydra:"
        )

        assert (
            len(hydra_values) == 0
        ), "hydra: values can not be resolved here. Set them manually in this fixture for tests!"

        OmegaConf.resolve(config)
        return config


def cleanup(config: DictConfig):
    # Cleanup: remove temporary dir
    if os.path.exists(config.path.run_dir):
        shutil.rmtree(config.path.run_dir)


@pytest.fixture
def classification_config() -> Generator[DictConfig, None, None]:
    _config = config("classification")
    yield _config
    cleanup(_config)


@pytest.fixture
def rl_config() -> Generator[DictConfig, None, None]:
    _config = config("gathering-apples")
    yield _config
    cleanup(_config)


@pytest.fixture
def data_root() -> str:
    return "cache"
