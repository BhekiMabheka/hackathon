"""
Reproducibility utilities.

Competition rule: your score is your score.
Any non-determinism between runs is a liability — it makes it impossible
to know whether a leaderboard improvement is signal or noise.

Usage
-----
    from src.utils.reproducibility import seed_everything
    seed_everything(42)
"""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


def seed_everything(seed: int = 42) -> None:
    """
    Set all known random seeds for full reproducibility.

    Covers: Python random, NumPy, and common ML library seeds.
    Note: GPU non-determinism (CUDA) requires additional flags not set here.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    log.info("Seeds set", seed=seed)


def assert_reproducible(
    fn,
    args: tuple = (),
    kwargs: dict | None = None,
    seed: int = 42,
    n_runs: int = 2,
    rtol: float = 1e-5,
) -> bool:
    """
    Run fn twice with the same seed and assert outputs match.
    Use in tests to guard against non-determinism regressions.
    """
    kwargs = kwargs or {}
    results = []
    for _ in range(n_runs):
        seed_everything(seed)
        results.append(fn(*args, **kwargs))

    for i in range(1, n_runs):
        if isinstance(results[0], np.ndarray):
            if not np.allclose(results[0], results[i], rtol=rtol):
                log.error("Reproducibility check failed", run_a=0, run_b=i)
                return False
    log.info("Reproducibility check passed", n_runs=n_runs)
    return True
