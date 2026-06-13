#!/usr/bin/env python3
"""Generate the defense-deck figures that are not yet exported to artifacts/figures/.

Two families of figures are produced:

1. Data figures (seaborn "whitegrid"/"talk" theme, to match the existing chart
   family such as ``dimensionality_saturation.png`` and ``expected_profit_by_scenario.png``):
     - ``score_distributions_new_vs_repeat.png``  — the simulator's actual per-segment
       credit-score distributions (base market). NOTE: the simulator draws scores from a
       *clipped Gaussian*, not a Beta; the curves below are sampled directly from the
       simulator's own generator, so they are not hand-drawn.
     - ``default_probability_mapping.png``        — the simulator's logistic score→PD curve
       for new vs. repeat clients, taken from ``build_market_state(...)['segment_params']``.

2. Schematic diagrams, re-exported from the existing thesis-source generator
   ``scripts/generate_simulator_diagrams.py`` (so they stay byte-for-byte the thesis
   figures) but written into ``artifacts/figures/`` under the deck's target filenames:
     - ``episode_timeline.png``           (<- episode_timeline.png)
     - ``delayed_reward_attribution.png`` (<- delayed_reward.png)
     - ``observation_layers.png``         (<- state_dimension_layers.png)
     - ``within_week_step.png``           (<- weekly_flowchart.png)
     - ``single_loan_lifecycle.png``      (<- loan_lifecycle.png)
   plus a small conceptual loop ``intro_concept.png`` for the intro slide.

All numbers come from the repo's own config/simulator code — nothing is fabricated.

Usage:
    python scripts/generate_defense_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent
SRC_DIR = SIM_ROOT / "src"
FIG_DIR = SIM_ROOT / "artifacts" / "figures"
for p in (str(SRC_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rl_credit_scoring_sim.env.scenarios import build_market_state  # noqa: E402
import generate_simulator_diagrams as gsd  # noqa: E402  (sibling script in scripts/)

DPI = 200

# New = blue, repeat = red — the exact convention used by plots.py for the
# new/repeat threshold paths, so the deck reads as one colour family.
COLOR_NEW = "#1f77b4"
COLOR_REPEAT = "#d62728"

# Policy facts straight from configs/run_profile.yaml (shared.policy).
THRESHOLD_MIN = 35.0
THRESHOLD_MAX = 85.0
DEFAULT_THRESHOLD_NEW = 60.0
DEFAULT_THRESHOLD_REPEAT = 50.0

# base_market scenario config (configs/scenarios.yaml) — only the keys
# build_market_state actually reads at week 0.
BASE_MARKET_CFG = {
    "repeat_share_base": 0.35,
    "weekly_volume_trend": 0.0,
    "seasonal_amplitude": 0.05,
    "score_shift_new": 0.0,
    "score_shift_repeat": 0.0,
    "default_shift_new": 0.0,
    "default_shift_repeat": 0.0,
    "score_noise_scale": 1.0,
    "volume_noise_scale": 0.08,
    "loan_amount_multiplier": 1.0,
    "recovery_shift": 0.0,
    "shock_start_week": None,
    "shock_magnitude": 0.0,
}
HORIZON_WEEKS = 26


def _base_market_segment_params() -> dict:
    """Pull the *actual* simulator segment parameters for base market, week 0."""
    state = build_market_state(
        scenario_name="base_market",
        scenario_cfg=BASE_MARKET_CFG,
        week_index=0,
        horizon_weeks=HORIZON_WEEKS,
    )
    return state["segment_params"], state["market"]


# ---------------------------------------------------------------------------
# Data figure 1 — score distributions (new vs. repeat)
# ---------------------------------------------------------------------------

def make_score_distributions() -> Path:
    import seaborn as sns

    segment_params, market = _base_market_segment_params()
    noise_scale = market["score_noise_scale"]
    rng = np.random.default_rng(0)
    n = 300_000

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for seg, color, label in (
        ("new", COLOR_NEW, "New clients"),
        ("repeat", COLOR_REPEAT, "Repeat clients"),
    ):
        p = segment_params[seg]
        # Sampled exactly as SyntheticCreditSimulator._generate_segment_batch does.
        scores = rng.normal(p["score_mean"], p["score_std"] * noise_scale, size=n)
        scores = np.clip(scores, 0.0, 100.0)
        sns.kdeplot(
            x=scores, ax=ax, fill=True, alpha=0.25, linewidth=2.4,
            color=color, clip=(0.0, 100.0),
            label=f"{label}  (μ={p['score_mean']:.0f}, σ={p['score_std']:.0f})",
        )
        ax.axvline(p["score_mean"], color=color, ls="--", lw=1.4, alpha=0.8)

    # Controllable threshold range + default thresholds (config facts).
    ax.axvspan(THRESHOLD_MIN, THRESHOLD_MAX, color="#888888", alpha=0.07, zorder=0)
    ax.axvline(DEFAULT_THRESHOLD_NEW, color=COLOR_NEW, ls=":", lw=1.6, alpha=0.7)
    ax.axvline(DEFAULT_THRESHOLD_REPEAT, color=COLOR_REPEAT, ls=":", lw=1.6, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ax.text(DEFAULT_THRESHOLD_NEW + 0.6, ymax * 0.96, "default θ_new = 60",
            color=COLOR_NEW, fontsize=10, rotation=90, va="top")
    ax.text(DEFAULT_THRESHOLD_REPEAT + 0.6, ymax * 0.96, "default θ_repeat = 50",
            color=COLOR_REPEAT, fontsize=10, rotation=90, va="top")
    ax.text((THRESHOLD_MIN + THRESHOLD_MAX) / 2, ymax * 0.02,
            "controllable threshold range [35, 85]", color="#555555",
            fontsize=9.5, ha="center", va="bottom", style="italic")

    ax.set_title("Credit-score distributions: new vs. repeat applicants (base market)")
    ax.set_xlabel("Credit score (0–100)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    out = FIG_DIR / "score_distributions_new_vs_repeat.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(SIM_ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Data figure 2 — score -> default-probability mapping
# ---------------------------------------------------------------------------

def make_default_probability_mapping() -> Path:
    import seaborn as sns  # noqa: F401  (kept so the theme is active)

    segment_params, _ = _base_market_segment_params()
    scores = np.linspace(0.0, 100.0, 501)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for seg, color, label in (
        ("new", COLOR_NEW, "New clients"),
        ("repeat", COLOR_REPEAT, "Repeat clients"),
    ):
        p = segment_params[seg]
        # default_probability_fn is the simulator's own logistic; clip mirrors
        # _generate_segment_batch's np.clip(default_prob, 0.01, 0.98).
        pd_vals = np.clip(
            np.array([p["default_probability_fn"](s) for s in scores]), 0.01, 0.98
        )
        ax.plot(scores, pd_vals * 100.0, color=color, lw=2.6, label=label)

    ax.axvspan(THRESHOLD_MIN, THRESHOLD_MAX, color="#888888", alpha=0.07, zorder=0)
    ax.set_title("Score → default-probability mapping (logistic, base market)")
    ax.set_xlabel("Credit score (0–100)")
    ax.set_ylabel("Default probability (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", framealpha=0.9)
    out = FIG_DIR / "default_probability_mapping.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(SIM_ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Schematic re-export — reuse the thesis-source generator, redirect output
# ---------------------------------------------------------------------------

# generator filename -> deck target filename
SCHEMATIC_NAME_MAP = {
    "episode_timeline.png": "episode_timeline.png",
    "delayed_reward.png": "delayed_reward_attribution.png",
    "state_dimension_layers.png": "observation_layers.png",
    "weekly_flowchart.png": "within_week_step.png",
    "loan_lifecycle.png": "single_loan_lifecycle.png",
}


def export_schematics() -> list[Path]:
    written: list[Path] = []

    def _save_override(fig, name):
        target = FIG_DIR / SCHEMATIC_NAME_MAP[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(target)
        print(f"  saved -> {target.relative_to(SIM_ROOT)}  (re-export of {name})")

    gsd._save = _save_override  # monkeypatch the module-global used by every diagram
    gsd.diagram_episode_timeline()
    gsd.diagram_delayed_reward()
    gsd.diagram_state_layers()
    gsd.diagram_weekly_flowchart()
    gsd.diagram_loan_lifecycle()
    return written


# ---------------------------------------------------------------------------
# intro_concept — weekly threshold-control loop (uses the schematic palette)
# ---------------------------------------------------------------------------

def make_intro_concept() -> Path:
    P = gsd.PALETTE
    fig, ax = plt.subplots(figsize=(14, 4.4), facecolor=P["bg"])
    ax.set_facecolor(P["bg"])
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.4, 4.2)
    ax.axis("off")

    ax.text(7.5, 3.95, "Weekly portfolio-level threshold control",
            ha="center", va="center", fontsize=15, fontweight="bold", color=P["text"])

    box_w, box_h, y = 2.78, 1.25, 1.6
    step = 3.005  # 5 boxes of 2.78 + 4 gaps of 0.225 span [0.1, 14.9]
    x0 = 0.1
    nodes = [
        (P["layer_a"], "Observe weekly\nstate (12–50D)", P["text"], 10.5),
        (P["layer_b"], "Choose thresholds\nθ_new , θ_repeat", P["text"], 10.5),
        (P["interactive"], "Accept loans\nscore ≥ θ", "white", 10.5),
        (P["layer_c"], "Delayed outcomes\nprofit · default · recovery", "white", 9.3),
        (P["layer_d"], "Shaped\nreward", "white", 10.5),
    ]
    centers = []
    for i, (color, label, tc, fs) in enumerate(nodes):
        x = x0 + i * step
        gsd._rounded_box(ax, x, y, box_w, box_h, color, label,
                         fontsize=fs, text_color=tc, radius=0.08)
        centers.append((x + box_w / 2, x, x + box_w))

    for i in range(len(nodes) - 1):
        gsd._arrow(ax, centers[i][2], y + box_h / 2, centers[i + 1][1], y + box_h / 2, lw=2.0)

    # feedback arrow: reward -> observe (next week)
    x_last_c = centers[-1][0]
    x_first_c = centers[0][0]
    ax.annotate(
        "", xy=(x_first_c, y - 0.02), xytext=(x_last_c, y - 0.02),
        arrowprops=dict(arrowstyle="-|>", color=P["accent"], lw=2.0,
                        connectionstyle="arc3,rad=0.32"), zorder=2,
    )
    ax.text(7.5, -0.2, "feedback loop — repeated every interactive week",
            ha="center", va="center", fontsize=10, color=P["accent"],
            fontweight="bold", style="italic")

    out = FIG_DIR / "intro_concept.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {out.relative_to(SIM_ROOT)}")
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating defense-deck figures ...")

    # Schematic diagrams + intro concept use matplotlib defaults / the gsd palette.
    plt.rcdefaults()
    print("[schematics]")
    export_schematics()
    make_intro_concept()

    # Data figures use the seaborn theme to match the existing chart family.
    print("[data figures]")
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    make_score_distributions()
    make_default_probability_mapping()

    print("Done.")


if __name__ == "__main__":
    main()
