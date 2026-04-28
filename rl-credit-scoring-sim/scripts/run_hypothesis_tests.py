from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DIMS = [12, 20, 30, 50]


def _load_run_level() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dim in DIMS:
        frame = pd.read_csv(OUTPUTS_DIR / f"dim_{dim}" / "run_level_metrics.csv")
        frame["state_dim"] = dim
        frame["pair_seed"] = frame["seed"] % 10000
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _load_weekly() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dim in DIMS:
        frame = pd.read_csv(OUTPUTS_DIR / f"dim_{dim}" / "weekly_run_metrics.csv")
        frame["state_dim"] = dim
        frame["pair_seed"] = frame["seed"] % 10000
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(values: np.ndarray, seed: int = 20260426) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.nan, np.nan
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    samples = rng.choice(clean, size=(5000, clean.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _paired_run_comparison(
    frame: pd.DataFrame,
    *,
    hypothesis: str,
    comparison: str,
    left_dim: int,
    left_controller: str,
    right_dim: int,
    right_controller: str,
    metric: str,
    scenario_name: str | None = None,
) -> dict[str, float | int | str]:
    left = frame[
        (frame["state_dim"] == left_dim) & (frame["controller"] == left_controller)
    ].copy()
    right = frame[
        (frame["state_dim"] == right_dim) & (frame["controller"] == right_controller)
    ].copy()
    if scenario_name is not None:
        left = left[left["scenario_name"] == scenario_name]
        right = right[right["scenario_name"] == scenario_name]

    join_cols = ["scenario_name", "pair_seed", "run_id"]
    merged = left[join_cols + [metric]].merge(
        right[join_cols + [metric]],
        on=join_cols,
        suffixes=("_left", "_right"),
    )
    diff = (merged[f"{metric}_left"] - merged[f"{metric}_right"]).to_numpy(float)
    diff = diff[np.isfinite(diff)]
    t_stat, p_value = stats.ttest_1samp(diff, 0.0) if diff.size > 1 else (np.nan, np.nan)
    if diff.size > 1 and np.any(np.abs(diff) > 1e-12):
        try:
            wilcoxon_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            wilcoxon_p = np.nan
    else:
        wilcoxon_p = np.nan
    ci_low, ci_high = _bootstrap_ci(diff)
    return {
        "hypothesis": hypothesis,
        "comparison": comparison,
        "scenario_name": scenario_name or "all",
        "metric": metric,
        "left": f"{left_controller}_{left_dim}D",
        "right": f"{right_controller}_{right_dim}D",
        "n_pairs": int(diff.size),
        "mean_left_minus_right": float(diff.mean()) if diff.size else np.nan,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "paired_t_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "wilcoxon_p_value": wilcoxon_p,
    }


def _weekly_gap_comparison(
    frame: pd.DataFrame,
    *,
    hypothesis: str,
    comparison: str,
    state_dim: int,
    controller: str,
    baseline: str,
    scenario_name: str,
    interactive_week: int,
) -> dict[str, float | int | str]:
    left = frame[
        (frame["state_dim"] == state_dim)
        & (frame["controller"] == controller)
        & (frame["scenario_name"] == scenario_name)
        & (frame["interactive_week"] == interactive_week)
    ].copy()
    right = frame[
        (frame["state_dim"] == state_dim)
        & (frame["controller"] == baseline)
        & (frame["scenario_name"] == scenario_name)
        & (frame["interactive_week"] == interactive_week)
    ].copy()
    join_cols = ["scenario_name", "pair_seed", "run_id", "interactive_week"]
    merged = left[join_cols + ["cumulative_profit"]].merge(
        right[join_cols + ["cumulative_profit"]],
        on=join_cols,
        suffixes=("_left", "_right"),
    )
    diff = (
        merged["cumulative_profit_left"] - merged["cumulative_profit_right"]
    ).to_numpy(float)
    diff = diff[np.isfinite(diff)]
    t_stat, p_value = stats.ttest_1samp(diff, 0.0) if diff.size > 1 else (np.nan, np.nan)
    ci_low, ci_high = _bootstrap_ci(diff)
    return {
        "hypothesis": hypothesis,
        "comparison": comparison,
        "scenario_name": scenario_name,
        "metric": f"cumulative_profit_week_{interactive_week}",
        "left": f"{controller}_{state_dim}D",
        "right": f"{baseline}_{state_dim}D",
        "n_pairs": int(diff.size),
        "mean_left_minus_right": float(diff.mean()) if diff.size else np.nan,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "paired_t_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "wilcoxon_p_value": float(stats.wilcoxon(diff).pvalue)
        if diff.size > 1 and np.any(np.abs(diff) > 1e-12)
        else np.nan,
    }


def _scenario_winner_counts() -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for dim in DIMS:
        frame = pd.read_csv(OUTPUTS_DIR / f"dim_{dim}" / "main_scenario_summary.csv")
        for metric, direction in [
            ("cumulative_reward_mean", "max"),
            ("expected_profit_mean", "max"),
            ("default_rate_mean", "min"),
        ]:
            if direction == "max":
                winners = frame.loc[frame.groupby("scenario_name")[metric].idxmax()]
            else:
                winners = frame.loc[frame.groupby("scenario_name")[metric].idxmin()]
            counts = winners["controller_type"].value_counts().to_dict()
            rows.append(
                {
                    "state_dim": dim,
                    "metric": metric,
                    "rl_scenario_wins": int(counts.get("agent", 0)),
                    "baseline_scenario_wins": int(counts.get("baseline", 0)),
                    "total_scenarios": int(winners.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    run_level = _load_run_level()
    weekly = _load_weekly()

    comparisons = [
        _paired_run_comparison(
            run_level,
            hypothesis="H1",
            comparison="30D DQN vs 12D DQN",
            left_dim=30,
            left_controller="dqn",
            right_dim=12,
            right_controller="dqn",
            metric="cumulative_reward",
        ),
        _paired_run_comparison(
            run_level,
            hypothesis="H1",
            comparison="30D DQN vs 20D Double-DQN best-RL frontier",
            left_dim=30,
            left_controller="dqn",
            right_dim=20,
            right_controller="double_dqn",
            metric="cumulative_reward",
        ),
        _paired_run_comparison(
            run_level,
            hypothesis="H1",
            comparison="30D DQN vs 50D DQN",
            left_dim=30,
            left_controller="dqn",
            right_dim=50,
            right_controller="dqn",
            metric="cumulative_reward",
        ),
        _paired_run_comparison(
            run_level,
            hypothesis="H1",
            comparison="30D DQN vs 20D Double-DQN best-RL frontier",
            left_dim=30,
            left_controller="dqn",
            right_dim=20,
            right_controller="double_dqn",
            metric="default_rate",
        ),
        _paired_run_comparison(
            run_level,
            hypothesis="H2",
            comparison="30D best RL vs best overall baseline",
            left_dim=30,
            left_controller="dqn",
            right_dim=30,
            right_controller="profit_oriented",
            metric="cumulative_reward",
        ),
        _paired_run_comparison(
            run_level,
            hypothesis="H2",
            comparison="50D best RL vs best overall baseline",
            left_dim=50,
            left_controller="dqn",
            right_dim=50,
            right_controller="profit_oriented",
            metric="cumulative_reward",
        ),
        _paired_run_comparison(
            run_level,
            hypothesis="H2",
            comparison="50D DQN vs profit-oriented baseline in base market",
            left_dim=50,
            left_controller="dqn",
            right_dim=50,
            right_controller="profit_oriented",
            metric="cumulative_reward",
            scenario_name="base_market",
        ),
        _paired_run_comparison(
            run_level,
            hypothesis="H2",
            comparison="30D DQN default rate vs profit-oriented baseline",
            left_dim=30,
            left_controller="dqn",
            right_dim=30,
            right_controller="profit_oriented",
            metric="default_rate",
        ),
        _weekly_gap_comparison(
            weekly,
            hypothesis="H3",
            comparison="30D DQN vs static threshold, adverse stress, early horizon",
            state_dim=30,
            controller="dqn",
            baseline="static_threshold",
            scenario_name="adverse_stress",
            interactive_week=6,
        ),
        _weekly_gap_comparison(
            weekly,
            hypothesis="H3",
            comparison="30D DQN vs static threshold, adverse stress, final horizon",
            state_dim=30,
            controller="dqn",
            baseline="static_threshold",
            scenario_name="adverse_stress",
            interactive_week=25,
        ),
    ]
    tests = pd.DataFrame(comparisons)
    tests.to_csv(OUTPUTS_DIR / "statistical_hypothesis_tests.csv", index=False)

    winner_counts = _scenario_winner_counts()
    winner_counts.to_csv(OUTPUTS_DIR / "scenario_winner_counts.csv", index=False)

    print(f"Wrote {OUTPUTS_DIR / 'statistical_hypothesis_tests.csv'}")
    print(f"Wrote {OUTPUTS_DIR / 'scenario_winner_counts.csv'}")


if __name__ == "__main__":
    main()
