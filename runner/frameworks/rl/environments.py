import os
from pathlib import Path

from runner.util import get_cache_dir


def available_envs() -> list[str]:
    """Return a list of available environments."""
    cache_dir = get_cache_dir()
    files = os.listdir(cache_dir / "scenarios")
    envs = [f.split(".")[0] for f in files if f.endswith(".zip")]
    envs.sort(reverse=True)  # Sort in reverse order to have the latest versions first
    return envs

def get_full_env_name(env_name: str) -> str:
    """Return the full environment name, including versioning if applicable."""
    available = available_envs()
    for env in available:
        if env.startswith(env_name):
            residual_env_name = env[len(env_name):]
            if residual_env_name == "" or residual_env_name.startswith("_v"):
                return env
    raise ValueError(f"Environment {env_name} not found in {get_cache_dir() / 'scenarios'}")