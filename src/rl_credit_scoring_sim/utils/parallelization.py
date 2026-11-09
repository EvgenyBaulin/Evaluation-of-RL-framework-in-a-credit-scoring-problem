"""
Parallel execution of the dimensionality experiment.

Two modes:

* ``run_dims_parallel``: each state dimension runs in its own subprocess.
* ``run_dim_seeds_parallel``: every (state dimension, seed) pair is trained and
  evaluated in its own subprocess, then each dimension's shards are merged in
  seed order and the tables and figures are built. This uses up to
  ``n_cpu_workers()`` cores instead of one core per dimension.

Both are safe because dimensions write to separate output directories and
checkpoint file names include the seed.
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


def _run_seed_shard(args: tuple) -> tuple[int, int, tuple | None, dict | None]:
    """Worker: train and evaluate all controllers for one (state_dim, seed)."""
    state_dim, seed, config, shared_scenarios = args
    try:
        from rl_credit_scoring_sim.evaluation.pipeline import run_seed_shard
        return state_dim, seed, run_seed_shard(config, shared_scenarios, seed), None
    except Exception as exc:  # noqa: BLE001
        return state_dim, seed, None, {
            "state_dim": state_dim,
            "error": f"seed {seed}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _finalize_dim(args: tuple) -> tuple[int, dict | None, dict | None]:
    """Worker: merge one dimension's shards and build its tables and figures."""
    state_dim, config, shared_scenarios, shards, build_artifacts_fn = args
    try:
        from rl_credit_scoring_sim.evaluation.pipeline import finalize_from_shards
        result = finalize_from_shards(config, shared_scenarios, shards)
        result["selection"] = build_artifacts_fn(result, state_dim)
        return state_dim, result, None
    except Exception as exc:  # noqa: BLE001
        return state_dim, None, {"state_dim": state_dim, "error": str(exc), "traceback": traceback.format_exc()}


def run_dim_seeds_parallel(
    dims: list[int],
    configs_by_dim: dict[int, dict[str, Any]],
    shared_scenarios: dict[str, Any],
    build_artifacts_fn: Callable,
    n_workers: int | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Run every (state_dim, seed) pair in its own process, then finalize per dimension.

    Returns the same structure as ``run_dims_parallel``.
    """
    n_workers = n_workers or n_cpu_workers()
    seeds_by_dim = {dim: list(configs_by_dim[dim]["seeds"]) for dim in dims}
    max_seeds = max(len(seeds) for seeds in seeds_by_dim.values())
    # Seed-major order so that all dimensions progress together.
    tasks = [
        (dim, seeds_by_dim[dim][i], configs_by_dim[dim], shared_scenarios)
        for i in range(max_seeds)
        for dim in dims
        if i < len(seeds_by_dim[dim])
    ]

    shards: dict[int, dict[int, tuple]] = {dim: {} for dim in dims}
    failures: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    # One task per child keeps memory flat and gives every shard a fresh process.
    with ctx.Pool(processes=min(n_workers, len(tasks)), maxtasksperchild=1) as pool:
        for state_dim, seed, shard, failure in pool.imap_unordered(_run_seed_shard, tasks):
            if failure is not None:
                failures.append(failure)
            else:
                shards[state_dim][seed] = shard

    failed_dims = {item["state_dim"] for item in failures}
    finalize_args = [
        (
            dim,
            configs_by_dim[dim],
            shared_scenarios,
            [shards[dim][seed] for seed in seeds_by_dim[dim]],
            build_artifacts_fn,
        )
        for dim in dims
        if dim not in failed_dims
    ]
    results: dict[int, dict[str, Any]] = {}
    if finalize_args:
        with ctx.Pool(processes=min(n_workers, len(finalize_args)), maxtasksperchild=1) as pool:
            for state_dim, result, failure in pool.map(_finalize_dim, finalize_args):
                if failure is not None:
                    failures.append(failure)
                else:
                    results[state_dim] = result
    return results, failures
