"""Env steps at which the RL analysis runs.

The schedule is 0, f, f + 2f, f + 2f + 4f, ... with the increment capped at
``end_freq`` (f = ``start_freq``). It only depends on the env steps, so a resumed
run can continue it where the interrupted run stopped.
"""


def advance(step: int, freq: int, end_freq: int) -> tuple[int, int]:
    """The analysis point after `step` and the frequency to use after that one."""
    return step + min(freq, end_freq), freq * 2


def first_after(env_steps: int, start_freq: int, end_freq: int) -> tuple[int, int]:
    """First analysis point strictly after `env_steps` (and its frequency),
    i.e. where a run resumed at `env_steps` continues the schedule."""
    step, freq = 0, start_freq
    while step <= env_steps:
        step, freq = advance(step, freq, end_freq)
    return step, freq
