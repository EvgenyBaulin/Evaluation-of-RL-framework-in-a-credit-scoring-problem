from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _apply_bar_error_bars(ax, plot_df: pd.DataFrame, value_col: str, lower_col: str, upper_col: str) -> None:
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        center_x = patch.get_x() + patch.get_width() / 2
        yerr = [
            [row[value_col] - row[lower_col]],
            [row[upper_col] - row[value_col]],
        ]
        ax.errorbar(
            x=center_x,
            y=row[value_col],
            yerr=yerr,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )


def plot_metric_bars(
    summary_df: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    mean_col = f"{metric}_mean"
    lower_col = f"{metric}_ci_lower"
    upper_col = f"{metric}_ci_upper"
    plot_df = summary_df.sort_values(["scenario_name", mean_col], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=plot_df,
        x="scenario_name",
        y=mean_col,
        hue="controller",
        ax=ax,
        palette="deep",
    )
    _apply_bar_error_bars(ax, plot_df, mean_col, lower_col, upper_col)
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=25)
    _save_figure(fig, output_path)


def plot_profit_bars(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_metric_bars(
        summary_df=summary_df,
        output_path=output_path,
        metric="expected_profit",
        title="Expected Profit by Scenario",
        y_label="Expected profit",
    )


def plot_cumulative_curves(curves_df: pd.DataFrame, output_path: Path, metric: str, title: str) -> None:
    scenarios = list(curves_df["scenario_name"].drop_duplicates())
    ncols = min(3, max(1, len(scenarios)))
    nrows = max(1, math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    mean_col = f"{metric}_mean"
    lower_col = f"{metric}_ci_lower"
    upper_col = f"{metric}_ci_upper"
    for ax, scenario_name in zip(axes, scenarios):
        scenario_df = curves_df[curves_df["scenario_name"] == scenario_name]
        for controller, controller_df in scenario_df.groupby("controller"):
            controller_df = controller_df.sort_values("interactive_week")
            ax.plot(controller_df["interactive_week"], controller_df[mean_col], label=controller)
            ax.fill_between(
                controller_df["interactive_week"],
                controller_df[lower_col],
                controller_df[upper_col],
                alpha=0.15,
            )
        ax.set_title(scenario_name)
        ax.set_xlabel("Interactive week")
        ax.set_ylabel(metric.replace("_", " ").title())
    for ax in axes[len(scenarios):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=min(4, len(labels)))
    fig.suptitle(title, y=1.02)
    _save_figure(fig, output_path)


def plot_threshold_paths(weekly_df: pd.DataFrame, output_path: Path, scenario_name: str, controllers: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    subset = weekly_df[(weekly_df["scenario_name"] == scenario_name) & (weekly_df["controller"].isin(controllers))]
    for metric, ax, label in [
        ("threshold_new", axes[0], "New-client threshold"),
        ("threshold_repeat", axes[1], "Repeat-client threshold"),
    ]:
        summary = (
            subset.groupby(["controller", "interactive_week"], as_index=False)[metric]
            .mean()
            .sort_values(["controller", "interactive_week"])
        )
        sns.lineplot(data=summary, x="interactive_week", y=metric, hue="controller", ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Interactive week")
        ax.set_ylabel("Threshold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=min(4, len(labels)))
    fig.suptitle(f"Threshold Paths in {scenario_name}", y=1.02)
    _save_figure(fig, output_path)


def plot_ablation_bars(ablation_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=ablation_summary, x="controller", y="cumulative_reward_mean", hue="scenario_name", ax=ax)
    ax.set_title("Ablation Impact on Cumulative Reward")
    ax.set_xlabel("Ablation")
    ax.set_ylabel("Cumulative reward")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_path)


def plot_locally_worse_globally_better(curve_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, group in curve_df.groupby("controller"):
        group = group.sort_values("interactive_week")
        ax.plot(group["interactive_week"], group["cumulative_profit_mean"], label=label)
        ax.fill_between(
            group["interactive_week"],
            group["cumulative_profit_ci_lower"],
            group["cumulative_profit_ci_upper"],
            alpha=0.15,
        )
    ax.set_title(title)
    ax.set_xlabel("Interactive week")
    ax.set_ylabel("Cumulative profit")
    ax.legend()
    _save_figure(fig, output_path)


def plot_best_rl_threshold_paths(
    weekly_df: pd.DataFrame,
    output_path: Path,
    controller_name: str,
    title: str,
) -> None:
    subset = weekly_df[weekly_df["controller"] == controller_name].copy()
    scenarios = list(subset["scenario_name"].drop_duplicates())
    ncols = min(3, max(1, len(scenarios)))
    nrows = max(1, math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, scenario_name in zip(axes, scenarios):
        scenario_df = subset[subset["scenario_name"] == scenario_name]
        if scenario_df.empty:
            ax.set_visible(False)
            continue
        summary = (
            scenario_df.groupby("interactive_week")
            .agg(
                threshold_new_mean=("threshold_new", "mean"),
                threshold_new_lower=("threshold_new", lambda values: values.quantile(0.025)),
                threshold_new_upper=("threshold_new", lambda values: values.quantile(0.975)),
                threshold_repeat_mean=("threshold_repeat", "mean"),
                threshold_repeat_lower=("threshold_repeat", lambda values: values.quantile(0.025)),
                threshold_repeat_upper=("threshold_repeat", lambda values: values.quantile(0.975)),
            )
            .reset_index()
        )
        ax.plot(summary["interactive_week"], summary["threshold_new_mean"], label="new", color="#1f77b4")
        ax.fill_between(
            summary["interactive_week"],
            summary["threshold_new_lower"],
            summary["threshold_new_upper"],
            alpha=0.15,
            color="#1f77b4",
        )
        ax.plot(summary["interactive_week"], summary["threshold_repeat_mean"], label="repeat", color="#d62728")
        ax.fill_between(
            summary["interactive_week"],
            summary["threshold_repeat_lower"],
            summary["threshold_repeat_upper"],
            alpha=0.15,
            color="#d62728",
        )
        ax.set_title(scenario_name)
        ax.set_xlabel("Interactive week")
        ax.set_ylabel("Threshold")

    for ax in axes[len(scenarios):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2)
    fig.suptitle(title, y=1.02)
    _save_figure(fig, output_path)


def plot_metric_vs_dimension(
    best_rl_df: pd.DataFrame,
    overall_best_df: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_col = f"{metric}_mean"
    lower_col = f"{metric}_ci_lower"
    upper_col = f"{metric}_ci_upper"

    for label, frame, color in [
        ("Best RL", best_rl_df.sort_values("state_dim"), "#1f77b4"),
        ("Best Overall", overall_best_df.sort_values("state_dim"), "#d62728"),
    ]:
        if frame.empty:
            continue
        ax.plot(frame["state_dim"], frame[mean_col], marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(frame["state_dim"], frame[lower_col], frame[upper_col], alpha=0.15, color=color)

    ax.set_title(title)
    ax.set_xlabel("State dimension")
    ax.set_ylabel(y_label)
    ax.set_xticks(sorted(set(best_rl_df["state_dim"]).union(set(overall_best_df["state_dim"]))))
    ax.legend()
    _save_figure(fig, output_path)
