"""Tests for retinal_rl/rl/sample_factory/analysis_schedule.py."""

import sys

import pytest

sys.path.append(".")
from retinal_rl.rl.sample_factory import analysis_schedule as schedule

START, END = 5, 20


def _points(n: int) -> list[int]:
    """The first n analysis points of a fresh run."""
    points, step, freq = [], 0, START
    for _ in range(n):
        points.append(step)
        step, freq = schedule.advance(step, freq, END)
    return points


def test_fresh_schedule():
    # increments 5, 10, 20 (capped), 20, ...
    assert _points(6) == [0, 5, 15, 35, 55, 75]


@pytest.mark.parametrize("env_steps", [0, 4, 5, 6, 15, 40, 1000])
def test_resume_continues_fresh_schedule(env_steps):
    step, freq = schedule.first_after(env_steps, START, END)
    points = _points(100)
    assert step == min(p for p in points if p > env_steps)
    # following points are the same as in an uninterrupted run
    following = [step]
    for _ in range(3):
        step, freq = schedule.advance(step, freq, END)
        following.append(step)
    i = points.index(following[0])
    assert following == points[i : i + 4]
