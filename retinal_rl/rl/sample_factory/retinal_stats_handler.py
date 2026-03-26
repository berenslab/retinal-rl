from typing import Dict

from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.misc import EPISODIC
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import static_vars


@static_vars(pickups=dict())
def retinal_stats_handler(_runner: Runner, msg: Dict, _policy_id: PolicyID) -> None:
    cur_episode_pickups = (
        msg[EPISODIC].get("episode_extra_stats", {}).get("pickup_counts", {})
    )
    for stat_key, stat_value in cur_episode_pickups.items():
        if stat_key not in retinal_stats_handler.pickups:
            retinal_stats_handler.pickups[stat_key] = 0
        if stat_value is not None:  # Only append if the stat value is not None
            retinal_stats_handler.pickups[stat_key] += stat_value
