"""Regenerate paper/figures/dimensionality_saturation.png from the curated
full-profile per-dimension summary (outputs/best_rl_by_dimension.csv and
outputs/overall_best_by_dimension.csv); used by paper/article.tex.

Usage:
    python scripts/regenerate_saturation_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
SRC_DIR = SIM_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_credit_scoring_sim.plotting.plots import plot_dimensionality_saturation  # noqa: E402

AGENT_LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double-DQN",
    "a3c": "A3C",
    "a2c": "A2C",
    "ppo": "PPO",
    "sac": "SAC",
}


def main() -> None:
    best_rl = pd.read_csv(SIM_ROOT / "outputs" / "best_rl_by_dimension.csv")
    best_overall = pd.read_csv(SIM_ROOT / "outputs" / "overall_best_by_dimension.csv")

    merged = pd.DataFrame({
        "state_dim": best_rl["state_dim"],
        "best_rl_profit": best_rl["expected_profit_mean"],
        "best_rl_default": best_rl["default_rate_mean"],
        "best_rl_agent": best_rl["controller"].map(AGENT_LABELS).fillna(best_rl["controller"]),
    }).merge(
        pd.DataFrame({
            "state_dim": best_overall["state_dim"],
            "best_overall_profit": best_overall["expected_profit_mean"],
            "best_overall_default": best_overall["default_rate_mean"],
        }),
        on="state_dim",
    )
    print(merged)

    out_path = SIM_ROOT / "paper" / "figures" / "dimensionality_saturation.png"
    plot_dimensionality_saturation(merged, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
