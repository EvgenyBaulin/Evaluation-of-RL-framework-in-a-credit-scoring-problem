from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: Iterable[float],
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_resamples, array.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return mean, lower, upper


def summarize_with_ci(
    frame: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    confidence_level: float,
    n_resamples: int,
    seed: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for metric in metric_cols:
            mean, lower, upper = bootstrap_ci(
                group[metric].to_numpy(),
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=seed,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_curve_with_ci(
    frame: pd.DataFrame,
    group_cols: list[str],
    x_col: str,
    y_col: str,
    confidence_level: float,
    n_resamples: int,
    seed: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for keys, group in frame.groupby(group_cols + [x_col], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols + [x_col], keys)}
        mean, lower, upper = bootstrap_ci(
            group[y_col].to_numpy(),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed,
        )
        row[f"{y_col}_mean"] = mean
        row[f"{y_col}_ci_lower"] = lower
        row[f"{y_col}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)
