from __future__ import annotations

import itertools

import numpy as np


def build_threshold_grid(policy_config: dict) -> np.ndarray:
    threshold_min = policy_config["threshold_min"]
    threshold_max = policy_config["threshold_max"]
    granularity = policy_config["threshold_granularity"]
    expected_size = policy_config["threshold_grid_size"]
    grid = np.arange(threshold_min, threshold_max + granularity, granularity, dtype=float)
    if grid.size != expected_size:
        raise ValueError(
            f"Threshold grid size mismatch: expected {expected_size}, got {grid.size}. "
            "Adjust threshold_min/threshold_max/threshold_granularity or threshold_grid_size."
        )
    return grid


def build_discrete_action_map(policy_config: dict) -> list[tuple[float, float]]:
    grid = build_threshold_grid(policy_config)
    if policy_config["split_policy"]:
        return [(float(new), float(repeat)) for new, repeat in itertools.product(grid, grid)]
    return [(float(value), float(value)) for value in grid]
