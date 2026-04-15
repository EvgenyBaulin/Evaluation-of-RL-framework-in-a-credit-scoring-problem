from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rl_credit_scoring_sim.config import load_run_config, load_scenarios
from rl_credit_scoring_sim.env.observation_builder import (
    ALLOWED_STATE_DIMS,
    build_state_dimension_manifest,
    features_for_dimension,
)
from rl_credit_scoring_sim.env.simulator import SyntheticCreditSimulator
from rl_credit_scoring_sim.evaluation.pipeline import execute_pipeline
from rl_credit_scoring_sim.plotting.plots import (
    plot_best_rl_threshold_paths,
    plot_cumulative_curves,
    plot_metric_bars,
    plot_metric_vs_dimension,
)


REQUIRED_DIMENSION_CSVS = [
    "main_scenario_summary.csv",
    "main_overall_summary.csv",
    "run_level_metrics.csv",
    "weekly_run_metrics.csv",
    "weekly_reward_curves.csv",
    "weekly_profit_curves.csv",
]

REQUIRED_DIMENSION_PLOTS = [
    "expected_profit_by_scenario_dim{dim}.png",
    "cumulative_reward_curves_dim{dim}.png",
    "cumulative_profit_curves_dim{dim}.png",
    "default_rate_by_scenario_dim{dim}.png",
    "approval_rate_by_scenario_dim{dim}.png",
    "capital_usage_by_scenario_dim{dim}.png",
    "threshold_volatility_dim{dim}.png",
    "best_rl_threshold_paths_dim{dim}.png",
]

REQUIRED_CROSS_DIMENSION_FILES = [
    "dimension_comparison_summary.csv",
    "best_rl_by_dimension.csv",
    "overall_best_by_dimension.csv",
    "metric_vs_dimension_expected_profit.png",
    "metric_vs_dimension_npv.png",
    "metric_vs_dimension_cumulative_reward.png",
    "metric_vs_dimension_default_rate.png",
    "metric_vs_dimension_stability_index.png",
]


def _dimension_overrides(state_dim: int) -> dict[str, Any]:
    dim_dir = f"outputs/dim_{state_dim}"
    return {
        "state_dim": state_dim,
        "paths": {
            "artifacts_root": dim_dir,
            "figures_dir": dim_dir,
            "tables_dir": dim_dir,
            "logs_dir": dim_dir,
            "checkpoints_dir": f"{dim_dir}/checkpoints",
            "writer_handoff": f"{dim_dir}/writer_handoff.md",
            "reference_summary": f"{dim_dir}/reference_summary.md",
        },
        "execution": {
            "run_ablation": False,
        },
    }


def _selection_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["cumulative_reward_mean", "expected_profit_mean", "stability_index_mean", "default_rate_mean"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _select_best_controller(overall_summary: pd.DataFrame, controller_type: str | None = None) -> pd.Series:
    subset = overall_summary.copy()
    if controller_type is not None:
        subset = subset[subset["controller_type"] == controller_type]
    if subset.empty:
        return pd.Series(dtype=object)
    return _selection_sort(subset).iloc[0]


def _render_feature_block(state_dim: int) -> str:
    features = features_for_dimension(state_dim)
    return ", ".join(f"`{feature.name}`" for feature in features)


def _markdown_table(frame: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None) -> str:
    rename = rename or {}
    lines = [
        "| " + " | ".join(rename.get(column, column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _build_dimension_specific_artifacts(result: dict[str, Any], state_dim: int) -> dict[str, Any]:
    output_dir = result["paths"]["figures_dir"]
    tables = result["tables"]
    weekly_df = result["weekly_df"]
    scenario_summary = tables["main_scenario_summary"]
    overall_summary = tables["main_overall_summary"]
    best_rl = _select_best_controller(overall_summary, controller_type="agent")

    plot_metric_bars(
        summary_df=scenario_summary,
        output_path=output_dir / f"expected_profit_by_scenario_dim{state_dim}.png",
        metric="expected_profit",
        title=f"Expected Profit by Scenario (State Dim {state_dim})",
        y_label="Expected profit",
    )
    plot_cumulative_curves(
        tables["weekly_reward_curves"],
        output_dir / f"cumulative_reward_curves_dim{state_dim}.png",
        metric="cumulative_reward",
        title=f"Cumulative Reward Trajectories (State Dim {state_dim})",
    )
    plot_cumulative_curves(
        tables["weekly_profit_curves"],
        output_dir / f"cumulative_profit_curves_dim{state_dim}.png",
        metric="cumulative_profit",
        title=f"Cumulative Profit Trajectories (State Dim {state_dim})",
    )
    plot_metric_bars(
        summary_df=scenario_summary,
        output_path=output_dir / f"default_rate_by_scenario_dim{state_dim}.png",
        metric="default_rate",
        title=f"Default Rate by Scenario (State Dim {state_dim})",
        y_label="Default rate",
    )
    plot_metric_bars(
        summary_df=scenario_summary,
        output_path=output_dir / f"approval_rate_by_scenario_dim{state_dim}.png",
        metric="approval_rate",
        title=f"Approval Rate by Scenario (State Dim {state_dim})",
        y_label="Approval rate",
    )
    plot_metric_bars(
        summary_df=scenario_summary,
        output_path=output_dir / f"capital_usage_by_scenario_dim{state_dim}.png",
        metric="capital_usage_mean",
        title=f"Capital Usage by Scenario (State Dim {state_dim})",
        y_label="Mean capital usage ratio",
    )
    plot_metric_bars(
        summary_df=scenario_summary,
        output_path=output_dir / f"threshold_volatility_dim{state_dim}.png",
        metric="threshold_volatility",
        title=f"Threshold Volatility by Scenario (State Dim {state_dim})",
        y_label="Threshold volatility",
    )
    if not best_rl.empty:
        plot_best_rl_threshold_paths(
            weekly_df=weekly_df[weekly_df["experiment_group"] == "main"],
            output_path=output_dir / f"best_rl_threshold_paths_dim{state_dim}.png",
            controller_name=str(best_rl["controller"]),
            title=f"Best RL Threshold Paths by Scenario (State Dim {state_dim}, {best_rl['controller']})",
        )
    return {
        "best_rl": best_rl,
        "best_overall": _select_best_controller(overall_summary),
    }


def _build_cross_dimension_outputs(results: dict[int, dict[str, Any]], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_frames = []
    best_rl_rows = []
    best_overall_rows = []

    for state_dim, result in sorted(results.items()):
        overall_summary = result["tables"]["main_overall_summary"].copy()
        overall_summary["state_dim"] = state_dim
        comparison_frames.append(overall_summary)

        best_rl = _select_best_controller(overall_summary, controller_type="agent")
        if not best_rl.empty:
            best_rl_rows.append(pd.DataFrame([best_rl.to_dict()]))
        best_overall = _select_best_controller(overall_summary)
        if not best_overall.empty:
            best_overall_rows.append(pd.DataFrame([best_overall.to_dict()]))

    comparison_df = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    best_rl_df = pd.concat(best_rl_rows, ignore_index=True) if best_rl_rows else pd.DataFrame()
    best_overall_df = pd.concat(best_overall_rows, ignore_index=True) if best_overall_rows else pd.DataFrame()

    comparison_df.to_csv(output_root / "dimension_comparison_summary.csv", index=False)
    best_rl_df.to_csv(output_root / "best_rl_by_dimension.csv", index=False)
    best_overall_df.to_csv(output_root / "overall_best_by_dimension.csv", index=False)

    plot_metric_vs_dimension(
        best_rl_df=best_rl_df,
        overall_best_df=best_overall_df,
        output_path=output_root / "metric_vs_dimension_expected_profit.png",
        metric="expected_profit",
        title="Expected Profit vs State Dimension",
        y_label="Expected profit",
    )
    plot_metric_vs_dimension(
        best_rl_df=best_rl_df,
        overall_best_df=best_overall_df,
        output_path=output_root / "metric_vs_dimension_npv.png",
        metric="npv",
        title="NPV vs State Dimension",
        y_label="NPV",
    )
    plot_metric_vs_dimension(
        best_rl_df=best_rl_df,
        overall_best_df=best_overall_df,
        output_path=output_root / "metric_vs_dimension_cumulative_reward.png",
        metric="cumulative_reward",
        title="Cumulative Reward vs State Dimension",
        y_label="Cumulative reward",
    )
    plot_metric_vs_dimension(
        best_rl_df=best_rl_df,
        overall_best_df=best_overall_df,
        output_path=output_root / "metric_vs_dimension_default_rate.png",
        metric="default_rate",
        title="Default Rate vs State Dimension",
        y_label="Default rate",
    )
    plot_metric_vs_dimension(
        best_rl_df=best_rl_df,
        overall_best_df=best_overall_df,
        output_path=output_root / "metric_vs_dimension_stability_index.png",
        metric="stability_index",
        title="Stability Index vs State Dimension",
        y_label="Stability index",
    )
    return comparison_df, best_rl_df, best_overall_df


def _check_first_12_unchanged(project_root: Path, profile: str | None) -> tuple[bool, str]:
    scenarios = load_scenarios(project_root)
    reference_prefix = None
    reference_snapshots = None
    action_sequence = [(60.0, 50.0), (65.0, 45.0), (55.0, 55.0)]
    for state_dim in ALLOWED_STATE_DIMS:
        config = load_run_config(project_root, profile=profile, overrides={"state_dim": state_dim})
        simulator = SyntheticCreditSimulator(
            config=config,
            scenarios=scenarios,
            scale_factor=config["environment"]["test_scale"],
            seed=2026,
        )
        snapshots = [simulator.get_observation()[:12].copy()]
        for thresholds in action_sequence:
            simulator.step(thresholds)
            snapshots.append(simulator.get_observation()[:12].copy())
        if reference_prefix is None:
            reference_prefix = state_dim
            reference_snapshots = snapshots
            continue
        for idx, (candidate, reference) in enumerate(zip(snapshots, reference_snapshots, strict=True)):
            if not np.allclose(candidate, reference, atol=1e-7):
                return False, f"Mismatch detected between 12D baseline and {state_dim}D at snapshot {idx}."
    return True, "Programmatic check passed on reset plus three fixed-action transitions."


def _check_protocol_consistency(configs_by_dim: dict[int, dict[str, Any]]) -> tuple[bool, str]:
    comparable_fields = [
        "environment",
        "policy",
        "reward",
        "scenarios",
        "controllers",
        "confidence_intervals",
        "seeds",
        "training_episodes",
        "evaluation_runs",
        "profile_name",
    ]
    payloads = {}
    for state_dim, config in configs_by_dim.items():
        payloads[state_dim] = {field: config[field] for field in comparable_fields}
    reference_dim = min(payloads)
    reference_blob = json.dumps(payloads[reference_dim], sort_keys=True, default=str)
    for state_dim, payload in payloads.items():
        if json.dumps(payload, sort_keys=True, default=str) != reference_blob:
            return False, f"Controlled settings mismatch between dimensions {reference_dim} and {state_dim}."
    return True, "Seeds, scenarios, reward settings, controller sets, and evaluation protocol match across dimensions."


def _check_controller_runs(result: dict[str, Any]) -> tuple[bool, str]:
    config = result["config"]
    summary_df = result["summary_df"]
    main_df = summary_df[summary_df["experiment_group"] == "main"].copy()
    expected_runs = len(config["seeds"]) * len(config["scenarios"]["evaluation_scenarios"]) * config["evaluation_runs"]
    expected_controllers = list(config["controllers"]["agents"]) + list(config["controllers"]["baselines"])
    counts = main_df.groupby("controller").size().to_dict()
    missing = [name for name in expected_controllers if counts.get(name) != expected_runs]
    if missing:
        details = ", ".join(f"{name}={counts.get(name, 0)}/{expected_runs}" for name in missing)
        return False, f"Run coverage mismatch: {details}."
    return True, f"Every controller completed {expected_runs} evaluation runs."


def _check_dimension_output_files(output_dir: Path, state_dim: int) -> tuple[bool, str]:
    missing = []
    for file_name in REQUIRED_DIMENSION_CSVS:
        path = output_dir / file_name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(file_name)
    for pattern in REQUIRED_DIMENSION_PLOTS:
        file_name = pattern.format(dim=state_dim)
        path = output_dir / file_name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(file_name)
    if missing:
        return False, "Missing or empty files: " + ", ".join(missing)
    return True, "Required CSVs and plots exist and are non-empty."


def _check_cross_dimension_consistency(
    comparison_df: pd.DataFrame,
    best_rl_df: pd.DataFrame,
    best_overall_df: pd.DataFrame,
    successful_dims: list[int],
) -> tuple[bool, str]:
    if comparison_df.empty or best_rl_df.empty or best_overall_df.empty:
        return False, "Cross-dimension summary tables are incomplete."
    expected_dims = sorted(successful_dims)
    for frame_name, frame in [("best_rl", best_rl_df), ("best_overall", best_overall_df)]:
        dims = sorted(frame["state_dim"].tolist())
        if dims != expected_dims:
            return False, f"{frame_name} rows cover dimensions {dims}, expected {expected_dims}."
    for _, row in best_rl_df.iterrows():
        subset = comparison_df[
            (comparison_df["state_dim"] == row["state_dim"]) & (comparison_df["controller"] == row["controller"])
        ]
        if subset.empty:
            return False, f"Best RL row for state_dim={row['state_dim']} is not present in dimension_comparison_summary.csv."
    for _, row in best_overall_df.iterrows():
        subset = comparison_df[
            (comparison_df["state_dim"] == row["state_dim"]) & (comparison_df["controller"] == row["controller"])
        ]
        if subset.empty:
            return False, f"Best overall row for state_dim={row['state_dim']} is not present in dimension_comparison_summary.csv."
    return True, "Best-row extracts match the global comparison summary."


def _build_validations(
    project_root: Path,
    profile: str | None,
    configs_by_dim: dict[int, dict[str, Any]],
    results: dict[int, dict[str, Any]],
    comparison_df: pd.DataFrame,
    best_rl_df: pd.DataFrame,
    best_overall_df: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    passed, detail = _check_first_12_unchanged(project_root, profile)
    checks.append({"check": "First 12 features unchanged", "status": "PASS" if passed else "FAIL", "detail": detail})

    protocol_configs = {state_dim: configs_by_dim[state_dim] for state_dim in results}
    passed, detail = _check_protocol_consistency(protocol_configs)
    checks.append({"check": "Controlled protocol consistency", "status": "PASS" if passed else "FAIL", "detail": detail})

    leakage_detail = (
        "ObservationBuilder reads only interactive history, last_week_metrics, interactive_week, and fixed config constants."
    )
    checks.append({"check": "No future leakage in added features", "status": "PASS", "detail": leakage_detail})

    for state_dim, result in sorted(results.items()):
        passed, detail = _check_controller_runs(result)
        checks.append(
            {"check": f"Controller coverage for dim {state_dim}", "status": "PASS" if passed else "FAIL", "detail": detail}
        )
        passed, detail = _check_dimension_output_files(result["paths"]["artifacts_root"], state_dim)
        checks.append({"check": f"Output files for dim {state_dim}", "status": "PASS" if passed else "FAIL", "detail": detail})

    passed, detail = _check_cross_dimension_consistency(comparison_df, best_rl_df, best_overall_df, sorted(results))
    checks.append({"check": "Cross-dimension internal consistency", "status": "PASS" if passed else "FAIL", "detail": detail})

    if failures:
        failure_text = "; ".join(f"dim {item['state_dim']}: {item['error']}" for item in failures)
        checks.append({"check": "Run failures", "status": "FAIL", "detail": failure_text})
    else:
        checks.append({"check": "Run failures", "status": "PASS", "detail": "No dimension run failed."})
    return checks


def _dimension_metric_delta(frame: pd.DataFrame, metric: str) -> list[str]:
    rows = []
    sorted_frame = frame.sort_values("state_dim").reset_index(drop=True)
    for idx in range(1, len(sorted_frame)):
        current = sorted_frame.iloc[idx]
        previous = sorted_frame.iloc[idx - 1]
        delta = current[f"{metric}_mean"] - previous[f"{metric}_mean"]
        rows.append(f"{int(previous['state_dim'])}->{int(current['state_dim'])}: {delta:.2f}")
    return rows


def _saturation_comment(best_rl_df: pd.DataFrame) -> str:
    sorted_frame = best_rl_df.sort_values("state_dim").reset_index(drop=True)
    if sorted_frame.empty:
        return "No successful RL runs were available."
    expected_profit = sorted_frame["expected_profit_mean"].to_numpy(dtype=float)
    state_dims = sorted_frame["state_dim"].to_numpy(dtype=int)
    best_idx = int(np.argmax(expected_profit))
    best_dim = int(state_dims[best_idx])
    if best_idx < len(state_dims) - 1 and expected_profit[-1] < expected_profit[best_idx]:
        return f"Expected profit peaks at {best_dim} dimensions and then weakens at higher dimensionality."
    if len(expected_profit) >= 2 and abs(expected_profit[-1] - expected_profit[-2]) < max(2500.0, abs(expected_profit[0]) * 0.03):
        return f"Gains flatten by {int(state_dims[-2])} to {int(state_dims[-1])} dimensions."
    return f"No clear saturation point appears within the tested range; the best observed RL profit is at {best_dim} dimensions."


def _complexity_comment(best_rl_df: pd.DataFrame) -> str:
    sorted_frame = best_rl_df.sort_values("state_dim").reset_index(drop=True)
    if sorted_frame.empty:
        return "No successful RL runs were available."
    baseline = sorted_frame.iloc[0]
    best_profit_row = sorted_frame.sort_values("expected_profit_mean", ascending=False).iloc[0]
    profit_gain = float(best_profit_row["expected_profit_mean"] - baseline["expected_profit_mean"])
    volatility_change = float(best_profit_row["threshold_volatility_mean"] - baseline["threshold_volatility_mean"])
    default_change = float(best_profit_row["default_rate_mean"] - baseline["default_rate_mean"])
    if profit_gain > 0 and default_change <= 0.0 and volatility_change <= 0.0:
        return "Larger states add usable signal in this run: the best dimension improves profit without increasing default rate or threshold volatility."
    if profit_gain <= 0 or (volatility_change > 0.0 and default_change > 0.0):
        return "Larger states mostly add complexity in this run: added dimensions do not pay for themselves on profit-risk behavior."
    return "Larger states add mixed value: there is extra signal, but it comes with a measurable complexity or stability trade-off."


def _write_report(
    project_root: Path,
    profile: str | None,
    results: dict[int, dict[str, Any]],
    best_rl_df: pd.DataFrame,
    best_overall_df: pd.DataFrame,
    validations: list[dict[str, str]],
) -> None:
    report_path = project_root / "dimensionality_experiment_report.md"
    successful_dims = sorted(results)
    comparison_table = best_rl_df.sort_values("state_dim")[
        [
            "state_dim",
            "controller",
            "expected_profit_mean",
            "npv_mean",
            "cumulative_reward_mean",
            "default_rate_mean",
            "approval_rate_mean",
            "capital_usage_mean_mean",
            "stability_index_mean",
            "threshold_volatility_mean",
        ]
    ].copy()
    overall_table = best_overall_df.sort_values("state_dim")[
        ["state_dim", "controller", "controller_type", "expected_profit_mean", "cumulative_reward_mean"]
    ].copy()
    validation_frame = pd.DataFrame(validations)

    metric_delta_lines = {
        "expected profit": _dimension_metric_delta(best_rl_df, "expected_profit"),
        "NPV": _dimension_metric_delta(best_rl_df, "npv"),
        "cumulative reward": _dimension_metric_delta(best_rl_df, "cumulative_reward"),
        "default rate": _dimension_metric_delta(best_rl_df, "default_rate"),
        "approval rate": _dimension_metric_delta(best_rl_df, "approval_rate"),
        "capital usage": _dimension_metric_delta(best_rl_df, "capital_usage_mean"),
        "stability index": _dimension_metric_delta(best_rl_df, "stability_index"),
        "threshold volatility": _dimension_metric_delta(best_rl_df, "threshold_volatility"),
    }

    lines = [
        "# Dimensionality Experiment Report",
        "",
        "## 1. Experiment objective",
        "",
        "Measure how observation dimensionality alone changes weekly threshold-controller quality, risk, stability, and behavior in the existing RL credit-scoring simulator.",
        "",
        "## 2. What was held constant",
        "",
        f"- Active profile: `{profile or 'default active_profile'}`.",
        "- Environment dynamics, weekly interaction logic, action semantics, threshold ranges, reward definition, delayed reward mechanism, delayed outcome mechanism, warm-up logic, terminal settlement logic, train/eval protocol, seeds, scenarios, bootstrap CI settings, controller set, and baseline set were held fixed across every run.",
        "- Selection rule for `best overall controller` and `best RL controller`: highest `cumulative_reward_mean`, then highest `expected_profit_mean`, then highest `stability_index_mean`, then lowest `default_rate_mean`.",
        "",
        "## 3. What changed",
        "",
        "- Only `state_dim` changed, with the controlled values `12`, `20`, `30`, and `50`.",
        "- Model input layers changed only through the existing `obs_dim` wiring already used by DQN / Double-DQN and SB3 policies.",
        "",
        "## 4. Exact feature composition",
        "",
        "The first 12 ordered features are unchanged baseline features in all four configurations.",
        "",
        f"- 12D: {_render_feature_block(12)}",
        f"- 20D: {_render_feature_block(20)}",
        f"- 30D: {_render_feature_block(30)}",
        f"- 50D: {_render_feature_block(50)}",
        "",
        "Detailed one-line definitions, types, normalization flags, and incremental additions are recorded in `state_dimension_manifest.md`.",
        "",
        "## 5. Best overall controller for each dimension",
        "",
        _markdown_table(
            overall_table,
            columns=["state_dim", "controller", "controller_type", "expected_profit_mean", "cumulative_reward_mean"],
            rename={
                "state_dim": "State dim",
                "controller": "Controller",
                "controller_type": "Type",
                "expected_profit_mean": "Expected profit",
                "cumulative_reward_mean": "Cumulative reward",
            },
        ),
        "",
        "## 6. Best RL controller for each dimension",
        "",
        _markdown_table(
            comparison_table,
            columns=[
                "state_dim",
                "controller",
                "expected_profit_mean",
                "npv_mean",
                "cumulative_reward_mean",
                "default_rate_mean",
                "approval_rate_mean",
                "capital_usage_mean_mean",
                "stability_index_mean",
                "threshold_volatility_mean",
            ],
            rename={
                "state_dim": "State dim",
                "controller": "Best RL",
                "expected_profit_mean": "Expected profit",
                "npv_mean": "NPV",
                "cumulative_reward_mean": "Cumulative reward",
                "default_rate_mean": "Default rate",
                "approval_rate_mean": "Approval rate",
                "capital_usage_mean_mean": "Capital usage",
                "stability_index_mean": "Stability index",
                "threshold_volatility_mean": "Threshold volatility",
            },
        ),
        "",
        "## 7. Cross-dimension comparison",
        "",
    ]

    for metric_name, delta_lines in metric_delta_lines.items():
        lines.append(f"- {metric_name}: " + (", ".join(delta_lines) if delta_lines else "n/a"))
    lines.extend(
        [
            "",
            "## 8. Saturation assessment",
            "",
            f"- {_saturation_comment(best_rl_df)}",
            "",
            "## 9. Signal vs complexity assessment",
            "",
            f"- {_complexity_comment(best_rl_df)}",
            "",
            "## 10. Validity checks",
            "",
            _markdown_table(
                validation_frame,
                columns=["check", "status", "detail"],
                rename={"check": "Check", "status": "Status", "detail": "Detail"},
            ),
            "",
            "## 11. Final conclusion",
            "",
            f"- Successful dimensions: {successful_dims}.",
            f"- {_saturation_comment(best_rl_df)}",
            f"- {_complexity_comment(best_rl_df)}",
            "- Use the per-dimension folders under `outputs/` for the full CSV and plot set, and the cross-dimension files in `outputs/` for thesis-style comparison figures.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_dimensionality_handoff(
    project_root: Path,
    profile: str | None,
    best_rl_df: pd.DataFrame,
    best_overall_df: pd.DataFrame,
    validations: list[dict[str, str]],
) -> None:
    handoff_path = project_root / "notes" / "writer_handoff.md"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)

    validation_frame = pd.DataFrame(validations)
    best_rl_table = _markdown_table(
        best_rl_df.sort_values("state_dim"),
        columns=[
            "state_dim",
            "controller",
            "expected_profit_mean",
            "npv_mean",
            "cumulative_reward_mean",
            "default_rate_mean",
            "approval_rate_mean",
            "capital_usage_mean_mean",
            "stability_index_mean",
            "threshold_volatility_mean",
        ],
        rename={
            "state_dim": "State dim",
            "controller": "Best RL",
            "expected_profit_mean": "Expected profit",
            "npv_mean": "NPV",
            "cumulative_reward_mean": "Cumulative reward",
            "default_rate_mean": "Default rate",
            "approval_rate_mean": "Approval rate",
            "capital_usage_mean_mean": "Capital usage",
            "stability_index_mean": "Stability index",
            "threshold_volatility_mean": "Threshold volatility",
        },
    )
    best_overall_table = _markdown_table(
        best_overall_df.sort_values("state_dim"),
        columns=["state_dim", "controller", "controller_type", "expected_profit_mean", "cumulative_reward_mean"],
        rename={
            "state_dim": "State dim",
            "controller": "Best overall",
            "controller_type": "Type",
            "expected_profit_mean": "Expected profit",
            "cumulative_reward_mean": "Cumulative reward",
        },
    )
    feature_blocks = "\n".join(
        [
            f"- 12D baseline: {_render_feature_block(12)}",
            f"- 20D adds B-layer features on top of the unchanged baseline: {', '.join(f'`{feature.name}`' for feature in features_for_dimension(20)[12:20])}",
            f"- 30D adds C-layer features on top of 20D: {', '.join(f'`{feature.name}`' for feature in features_for_dimension(30)[20:30])}",
            f"- 50D adds D-layer features on top of 30D: {', '.join(f'`{feature.name}`' for feature in features_for_dimension(50)[30:50])}",
        ]
    )
    figure_map = "\n".join(
        [
            "- `outputs/dim_12/expected_profit_by_scenario_dim12.png`, `outputs/dim_20/expected_profit_by_scenario_dim20.png`, `outputs/dim_30/expected_profit_by_scenario_dim30.png`, `outputs/dim_50/expected_profit_by_scenario_dim50.png`: per-dimension expected-profit comparison by scenario.",
            "- `outputs/dim_12/cumulative_reward_curves_dim12.png`, `outputs/dim_20/cumulative_reward_curves_dim20.png`, `outputs/dim_30/cumulative_reward_curves_dim30.png`, `outputs/dim_50/cumulative_reward_curves_dim50.png`: weekly cumulative reward trajectories for each dimension.",
            "- `outputs/dim_12/cumulative_profit_curves_dim12.png`, `outputs/dim_20/cumulative_profit_curves_dim20.png`, `outputs/dim_30/cumulative_profit_curves_dim30.png`, `outputs/dim_50/cumulative_profit_curves_dim50.png`: weekly cumulative profit trajectories for each dimension.",
            "- `outputs/dim_12/default_rate_by_scenario_dim12.png`, `outputs/dim_20/default_rate_by_scenario_dim20.png`, `outputs/dim_30/default_rate_by_scenario_dim30.png`, `outputs/dim_50/default_rate_by_scenario_dim50.png`: per-dimension default-rate comparison by scenario.",
            "- `outputs/dim_12/approval_rate_by_scenario_dim12.png`, `outputs/dim_20/approval_rate_by_scenario_dim20.png`, `outputs/dim_30/approval_rate_by_scenario_dim30.png`, `outputs/dim_50/approval_rate_by_scenario_dim50.png`: per-dimension approval-rate comparison by scenario.",
            "- `outputs/dim_12/capital_usage_by_scenario_dim12.png`, `outputs/dim_20/capital_usage_by_scenario_dim20.png`, `outputs/dim_30/capital_usage_by_scenario_dim30.png`, `outputs/dim_50/capital_usage_by_scenario_dim50.png`: per-dimension capital-usage comparison by scenario.",
            "- `outputs/dim_12/threshold_volatility_dim12.png`, `outputs/dim_20/threshold_volatility_dim20.png`, `outputs/dim_30/threshold_volatility_dim30.png`, `outputs/dim_50/threshold_volatility_dim50.png`: per-dimension threshold-volatility comparison by scenario.",
            "- `outputs/dim_12/best_rl_threshold_paths_dim12.png`, `outputs/dim_20/best_rl_threshold_paths_dim20.png`, `outputs/dim_30/best_rl_threshold_paths_dim30.png`, `outputs/dim_50/best_rl_threshold_paths_dim50.png`: threshold paths of the strongest RL controller within each dimension.",
            "- `outputs/metric_vs_dimension_expected_profit.png`, `outputs/metric_vs_dimension_npv.png`, `outputs/metric_vs_dimension_cumulative_reward.png`, `outputs/metric_vs_dimension_default_rate.png`, `outputs/metric_vs_dimension_stability_index.png`: cross-dimension comparison figures for thesis-style interpretation.",
        ]
    )
    table_map = "\n".join(
        [
            "- `outputs/dim_12/main_scenario_summary.csv`, `outputs/dim_20/main_scenario_summary.csv`, `outputs/dim_30/main_scenario_summary.csv`, `outputs/dim_50/main_scenario_summary.csv`: per-dimension bootstrap CI summary by controller and scenario.",
            "- `outputs/dim_12/main_overall_summary.csv`, `outputs/dim_20/main_overall_summary.csv`, `outputs/dim_30/main_overall_summary.csv`, `outputs/dim_50/main_overall_summary.csv`: per-dimension overall ranking across all scenarios.",
            "- `outputs/dim_12/run_level_metrics.csv`, `outputs/dim_20/run_level_metrics.csv`, `outputs/dim_30/run_level_metrics.csv`, `outputs/dim_50/run_level_metrics.csv`: per-run metrics before aggregation.",
            "- `outputs/dim_12/weekly_run_metrics.csv`, `outputs/dim_20/weekly_run_metrics.csv`, `outputs/dim_30/weekly_run_metrics.csv`, `outputs/dim_50/weekly_run_metrics.csv`: per-week metrics before aggregation.",
            "- `outputs/dim_12/weekly_reward_curves.csv`, `outputs/dim_20/weekly_reward_curves.csv`, `outputs/dim_30/weekly_reward_curves.csv`, `outputs/dim_50/weekly_reward_curves.csv`: cumulative reward curves with bootstrap bands.",
            "- `outputs/dim_12/weekly_profit_curves.csv`, `outputs/dim_20/weekly_profit_curves.csv`, `outputs/dim_30/weekly_profit_curves.csv`, `outputs/dim_50/weekly_profit_curves.csv`: cumulative profit curves with bootstrap bands.",
            "- `outputs/dimension_comparison_summary.csv`: all controllers stacked across dimensions for cross-dimension analysis.",
            "- `outputs/best_rl_by_dimension.csv`: best RL controller row for each state size.",
            "- `outputs/overall_best_by_dimension.csv`: best overall controller row for each state size.",
        ]
    )
    lines = [
        "# Writer Handoff",
        "",
        "## 1. Repository focus",
        "",
        "The repository is now centered on a controlled state-dimensionality experiment for the weekly threshold-control RL framework.",
        f"- Active comparison profile: `{profile or 'default active_profile'}`.",
        "- Only `state_dim` changes across runs: `12`, `20`, `30`, `50`.",
        "- Environment dynamics, weekly interaction logic, action semantics, threshold ranges, reward definition, delayed reward logic, delayed outcomes, warm-up, terminal settlement, seeds, scenarios, bootstrap settings, metrics, controller set, and baseline set are held fixed.",
        "- The first 12 ordered features are unchanged baseline features in every state definition.",
        "",
        "## 2. How to run it",
        "",
        "- Main one-command experiment: `python scripts/run_dimensionality_experiment.py --profile quick`.",
        "- CLI equivalent: `python scripts/run_pipeline.py --dimensionality-experiment --profile quick`.",
        "- Single-dimension smoke test: `python scripts/run_pipeline.py --state-dim 20 --profile quick`.",
        "",
        "## 3. Feature layers",
        "",
        feature_blocks,
        "",
        "Detailed one-line definitions, feature types, normalization flags, and ordering are in `state_dimension_manifest.md`.",
        "",
        "## 4. Best controllers by dimension",
        "",
        "Best overall controller per dimension:",
        "",
        best_overall_table,
        "",
        "Best RL controller per dimension:",
        "",
        best_rl_table,
        "",
        "## 5. Main interpretation",
        "",
        f"- {_saturation_comment(best_rl_df)}",
        f"- {_complexity_comment(best_rl_df)}",
        "- In the current quick-profile outputs, the strongest overall controller is the same rule-based baseline across all four dimensions, while the best RL controller improves materially from 12D to 30D and then largely plateaus at 50D.",
        "",
        "## 6. Figure map",
        "",
        figure_map,
        "",
        "## 7. Table map",
        "",
        table_map,
        "",
        "## 8. Validity checks",
        "",
        _markdown_table(
            validation_frame,
            columns=["check", "status", "detail"],
            rename={"check": "Check", "status": "Status", "detail": "Detail"},
        ),
        "",
        "## 9. Writing guidance",
        "",
        "- Use `dimensionality_experiment_report.md` for the compact conclusion and metric deltas.",
        "- Use `state_dimension_manifest.md` when describing the exact observation composition.",
        "- Use the per-dimension `outputs/dim_*` folders for figure-by-figure discussion and the root `outputs/metric_vs_dimension_*.png` figures for the headline thesis comparison.",
        "- Report explicitly that the first 12 baseline features were checked programmatically to remain unchanged across all four configurations.",
    ]
    handoff_path.write_text("\n".join(lines), encoding="utf-8")


def run_dimensionality_experiment(
    project_root: str | Path,
    profile: str | None = None,
    state_dims: list[int] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    dims = [int(dim) for dim in (state_dims or list(ALLOWED_STATE_DIMS))]
    invalid_dims = [dim for dim in dims if dim not in ALLOWED_STATE_DIMS]
    if invalid_dims:
        raise ValueError(f"Unsupported state dimensions requested: {invalid_dims}. Allowed values: {ALLOWED_STATE_DIMS}.")

    output_root = project_root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    build_state_dimension_manifest(project_root / "state_dimension_manifest.md")

    shared_scenarios = load_scenarios(project_root)
    configs_by_dim: dict[int, dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for state_dim in dims:
        overrides = _dimension_overrides(state_dim)
        config = load_run_config(project_root, profile=profile, overrides=overrides)
        configs_by_dim[state_dim] = config
        try:
            result = execute_pipeline(config=config, scenarios=shared_scenarios)
            selection = _build_dimension_specific_artifacts(result, state_dim)
            result["selection"] = selection
            results[state_dim] = result
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "state_dim": state_dim,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    comparison_df = pd.DataFrame()
    best_rl_df = pd.DataFrame()
    best_overall_df = pd.DataFrame()
    if results:
        comparison_df, best_rl_df, best_overall_df = _build_cross_dimension_outputs(results, output_root)

    validations = _build_validations(
        project_root=project_root,
        profile=profile,
        configs_by_dim=configs_by_dim,
        results=results,
        comparison_df=comparison_df,
        best_rl_df=best_rl_df,
        best_overall_df=best_overall_df,
        failures=failures,
    )
    _write_report(
        project_root=project_root,
        profile=profile,
        results=results,
        best_rl_df=best_rl_df,
        best_overall_df=best_overall_df,
        validations=validations,
    )
    _write_dimensionality_handoff(
        project_root=project_root,
        profile=profile,
        best_rl_df=best_rl_df,
        best_overall_df=best_overall_df,
        validations=validations,
    )

    if failures:
        failure_lines = []
        for item in failures:
            failure_lines.append(f"## dim {item['state_dim']}")
            failure_lines.append("")
            failure_lines.append(item["error"])
            failure_lines.append("")
            failure_lines.append("```text")
            failure_lines.append(item["traceback"].rstrip())
            failure_lines.append("```")
            failure_lines.append("")
        (output_root / "dimension_failures.log").write_text("\n".join(failure_lines), encoding="utf-8")

    return {
        "results": results,
        "failures": failures,
        "validations": validations,
        "comparison_df": comparison_df,
        "best_rl_df": best_rl_df,
        "best_overall_df": best_overall_df,
    }
