"""
Parallel execution of the dimensionality experiment.
Each state dimension runs in its own subprocess — safe because they write to
separate output directories and carry independent random state.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any, Callable

from rl_credit_scoring_sim.utils.device_utils import n_cpu_workers


def _run_one_dim(args: tuple) -> tuple[int, dict | None, dict | None]:
    """Worker: run execute_pipeline for a single state_dim."""
    state_dim, config, shared_scenarios, build_artifacts_fn = args
    try:
        from rl_credit_scoring_sim.evaluation.pipeline import execute_pipeline
        result = execute_pipeline(config=config, scenarios=shared_scenarios)
        selection = build_artifacts_fn(result, state_dim)
        result["selection"] = selection
        return state_dim, result, None
    except Exception as exc:  # noqa: BLE001
        return state_dim, None, {"state_dim": state_dim, "error": str(exc), "traceback": traceback.format_exc()}


def run_dims_parallel(
    dims: list[int],
    configs_by_dim: dict[int, dict[str, Any]],
    shared_scenarios: dict[str, Any],
    build_artifacts_fn: Callable,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Run each dimension in a separate process.

    Returns:
        results: {state_dim: pipeline_result}
        failures: list of failure dicts
    """
    n_workers = min(len(dims), n_cpu_workers())
    args = [
        (dim, configs_by_dim[dim], shared_scenarios, build_artifacts_fn)
        for dim in dims
    ]
    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    # Use spawn context for MPS/CUDA safety (avoids forked GPU state)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for state_dim, result, failure in pool.map(_run_one_dim, args):
            if failure is not None:
                failures.append(failure)
            else:
                results[state_dim] = result

    return results, failures
