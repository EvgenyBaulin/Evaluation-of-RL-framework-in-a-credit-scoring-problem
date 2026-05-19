"""
Generate conceptual simulator diagrams for the thesis.

Outputs are written to outputs/diagrams/ as high-resolution PNGs.

Usage:
    python scripts/generate_simulator_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).parent.parent / "outputs" / "diagrams"
DPI = 180

PALETTE = {
    "warmup":      "#b0c4de",
    "interactive": "#4a90d9",
    "terminal":    "#2c5f8a",
    "accent":      "#e8a838",
    "positive":    "#5aab61",
    "negative":    "#d95f5f",
    "neutral":     "#888888",
    "layer_a":     "#d0e8f5",
    "layer_b":     "#a8d0eb",
    "layer_c":     "#6fb0d8",
    "layer_d":     "#3a8fc0",
    "bg":          "#f8f9fa",
    "text":        "#1a1a2e",
    "arrow":       "#555577",
    "box_border":  "#334466",
}


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path.relative_to(Path(__file__).parent.parent.parent)}")


def _rounded_box(ax, x, y, w, h, color, label, sublabel=None,
                 fontsize=11, subsize=9, text_color="white", radius=0.04,
                 border_color=None, alpha=1.0):
    bc = border_color or PALETTE["box_border"]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=bc, linewidth=1.5, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    cy = y + h / 2 + (0.015 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.045, sublabel,
                ha="center", va="center", fontsize=subsize,
                color=text_color, alpha=0.88, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color=None, lw=1.8, style="-|>"):
    c = color or PALETTE["arrow"]
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=c, lw=lw),
        zorder=5,
    )


# ---------------------------------------------------------------------------
# 2. Episode timeline
# ---------------------------------------------------------------------------

def diagram_episode_timeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.8, 5.0)
    ax.axis("off")

    ax.text(7.5, 4.75, "Episode Structure", ha="center", va="center",
            fontsize=15, fontweight="bold", color=PALETTE["text"])

    # ── phase boxes ───────────────────────────────────────────────────────────
    phases = [
        (0.0,  3.2, PALETTE["warmup"],      "Warm-up phase"),
        (3.4,  7.2, PALETTE["interactive"], "Interactive phase"),
        (10.8, 3.5, PALETTE["terminal"],    "Terminal settlement"),
    ]
    for x, w, col, label in phases:
        box = FancyBboxPatch(
            (x, 1.5), w, 2.0,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=col, edgecolor=PALETTE["box_border"], linewidth=1.8, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, 1.5 + 2.0 / 2, label,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", zorder=4)

    # ── sub-labels below each box ────────────────────────────────────────────
    sublabels = [
        (1.6,  "Weeks 0–3\nNo agent action;\nhistory buffer fills"),
        (7.0,  "Weeks 4–55\nAgent selects (threshold_new, threshold_repeat)\neach week and receives a shaped reward"),
        (12.55,"Week 56+\nAll outstanding loans resolved\nto final profit & loss"),
    ]
    for sx, slbl in sublabels:
        ax.text(sx, 1.42, slbl, ha="center", va="top", fontsize=8.5,
                color=PALETTE["text"], linespacing=1.5)

    # ── timeline arrow ───────────────────────────────────────────────────────
    _arrow(ax, -0.3, 0.7, 15.3, 0.7, lw=2.2)
    ax.text(15.4, 0.7, "time", ha="left", va="center",
            fontsize=10, color=PALETTE["text"])

    # ── tick marks ───────────────────────────────────────────────────────────
    ticks = [
        (0.0,  "t = 0\n(episode start)"),
        (3.2,  "Warm-up ends"),
        (10.6, "Interactive\nphase ends"),
        (14.3, "All loans\nsettled"),
    ]
    for xp, lbl in ticks:
        ax.plot([xp, xp], [0.55, 0.87], color=PALETTE["text"], lw=1.5, zorder=4)
        ax.text(xp, 0.42, lbl, ha="center", va="top", fontsize=8,
                color=PALETTE["text"], linespacing=1.4)

    # ── callout annotation (positioned clear of title) ───────────────────────
    ax.annotate(
        "observe state → choose thresholds → receive reward",
        xy=(7.0, 3.5), xytext=(7.0, 4.35),
        ha="center", va="center", fontsize=9, color=PALETTE["text"],
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["neutral"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=PALETTE["neutral"], alpha=0.9),
    )

    _save(fig, "episode_timeline.png")


# ---------------------------------------------------------------------------
# 3. Single-loan lifecycle
# ---------------------------------------------------------------------------

def diagram_loan_lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    ax.text(7.5, 5.2, "Single-Loan Lifecycle", ha="center", va="center",
            fontsize=15, fontweight="bold", color=PALETTE["text"])

    # ── swim lanes ──────────────────────────────────────────────────────────
    lane_labels = ["Application\narrives", "Acceptance\ndecision", "Repayment\nwindow", "Loan outcome"]
    lane_ys     = [4.2, 3.1, 2.0, 0.8]
    lane_colors = [PALETTE["layer_a"], PALETTE["layer_b"], PALETTE["layer_c"], PALETTE["layer_d"]]

    for lbl, ly, lc in zip(lane_labels, lane_ys, lane_colors):
        ax.barh(ly, 15.0, left=0.0, height=0.7,
                color=lc, alpha=0.25, zorder=1)
        ax.text(-0.3, ly, lbl, ha="right", va="center", fontsize=9,
                color=PALETTE["text"], linespacing=1.4)

    # ── main nodes ──────────────────────────────────────────────────────────
    nodes = [
        (1.0,  4.2, PALETTE["layer_a"],   "Application\narrived\n(week t)"),
        (4.0,  3.1, PALETTE["layer_b"],   "Score ≥ threshold\n→ Accepted"),
        (4.0,  4.2, PALETTE["negative"],  "Score < threshold\n→ Rejected"),
        (7.5,  2.0, PALETTE["layer_c"],   "Loan outstanding\n(weeks t+1 … t+L)"),
        (11.5, 0.8, PALETTE["positive"],  "Repaid\n(profit realized)"),
        (11.5, 1.8, PALETTE["accent"],    "Default detected\n(loss + recovery)"),
        (11.5, 2.8, PALETTE["neutral"],   "Early\nprepayment"),
    ]
    for nx, ny, nc, nlbl in nodes:
        _rounded_box(ax, nx - 1.0, ny - 0.28, 2.0, 0.56, nc, nlbl,
                     fontsize=8.5, radius=0.05)

    # ── arrows ───────────────────────────────────────────────────────────────
    # application → accepted
    _arrow(ax, 2.0, 4.2, 3.0, 3.38)
    ax.text(2.25, 3.85, "score ≥ θ", fontsize=8, color=PALETTE["text"], rotation=20)

    # application → rejected
    _arrow(ax, 1.0, 4.48, 3.0, 4.24)
    ax.text(1.5, 4.55, "score < θ", fontsize=8, color=PALETTE["text"])

    # accepted → outstanding
    _arrow(ax, 5.0, 3.1, 6.5, 2.1)
    ax.text(5.4, 2.7, "loan disbursed\nweek t+1", fontsize=8, color=PALETTE["text"], linespacing=1.3)

    # outstanding → outcomes
    for oy in [0.8, 1.8, 2.8]:
        _arrow(ax, 8.5, 2.0, 10.5, oy)

    # ── repayment window bracket ─────────────────────────────────────────────
    ax.annotate("", xy=(10.4, 1.7), xytext=(6.5, 1.7),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"], lw=1.5))
    ax.text(8.45, 1.4, "L weeks of delayed outcome\n(reward arrives with lag)",
            ha="center", fontsize=8.5, color=PALETTE["accent"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE["accent"], alpha=0.8))

    # ── rejected annotation ──────────────────────────────────────────────────
    ax.text(4.0, 4.72, "no capital used;\nno future signal",
            ha="center", fontsize=8, color=PALETTE["negative"], style="italic")

    _save(fig, "loan_lifecycle.png")


# ---------------------------------------------------------------------------
# 4. Delayed reward attribution diagram
# ---------------------------------------------------------------------------

def diagram_delayed_reward() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.8), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1.2, 5.5)

    ax.text(6.5, 5.2, "Delayed Reward Attribution", ha="center", va="center",
            fontsize=15, fontweight="bold", color=PALETTE["text"])
    ax.text(6.5, 4.85, "Actions taken at week t produce portfolio signals spread over future weeks",
            ha="center", va="center", fontsize=10, color=PALETTE["neutral"])

    n_cohorts   = 5
    loan_life   = 5     # weeks a cohort stays "outstanding"
    cohort_cols = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]

    week_xs = np.arange(0, 13)
    week_labels = [f"Week {w}" for w in week_xs]

    # ── time axis ────────────────────────────────────────────────────────────
    ax.axhline(y=-0.3, xmin=0.02, xmax=0.97, color=PALETTE["text"], lw=1.5)
    for wx in week_xs:
        ax.plot([wx, wx], [-0.45, -0.18], color=PALETTE["text"], lw=1.2)
        ax.text(wx, -0.65, f"t+{wx}", ha="center", va="top", fontsize=8,
                color=PALETTE["text"])
    ax.text(-0.7, -0.3, "Time →", ha="left", va="center", fontsize=9,
            color=PALETTE["text"])

    # ── cohort bars ──────────────────────────────────────────────────────────
    lane_height = 0.55
    lane_gap    = 0.15
    bar_top     = 4.3

    for i, col in enumerate(cohort_cols):
        t_accept = i          # cohort accepted at week i
        t_end    = t_accept + loan_life

        y_bot = bar_top - i * (lane_height + lane_gap)
        y_ctr = y_bot + lane_height / 2

        # action marker (diamond)
        ax.scatter([t_accept], [y_ctr], marker="D", s=80, color=col,
                   edgecolors=PALETTE["box_border"], zorder=6, linewidths=1.2)
        ax.text(t_accept - 0.05, y_ctr + 0.32,
                f"Action at t+{i}", ha="center", fontsize=8,
                color=PALETTE["text"], fontweight="bold")

        # outstanding bar
        bar_x = np.linspace(t_accept + 0.15, min(t_end, 12.4), 200)
        ax.fill_between(bar_x, y_bot, y_bot + lane_height,
                        color=col, alpha=0.55, zorder=2)
        ax.plot([t_accept + 0.15, min(t_end, 12.4)], [y_bot, y_bot],
                color=col, lw=1, alpha=0.6)
        ax.plot([t_accept + 0.15, min(t_end, 12.4)], [y_bot + lane_height, y_bot + lane_height],
                color=col, lw=1, alpha=0.6)

        # outcome star
        if t_end <= 12:
            ax.scatter([t_end], [y_ctr], marker="*", s=140, color=PALETTE["accent"],
                       edgecolors="white", zorder=7, linewidths=0.8)

    # ── vertical "overlap" bands ──────────────────────────────────────────────
    for wx in [4, 5, 6, 7]:
        ax.axvspan(wx - 0.08, wx + 0.08, ymin=0.05, ymax=0.85,
                   color=PALETTE["accent"], alpha=0.12, zorder=1)
    ax.text(5.5, 0.25, "← Simultaneous contributions from\n   multiple overlapping cohorts",
            ha="center", fontsize=8.5, color=PALETTE["accent"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["accent"], alpha=0.85))

    # ── legend ───────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor="#6baed6", alpha=0.55, label="Loan outstanding (cohort bar)"),
        plt.scatter([], [], marker="D", s=60, color="#888", label="Threshold action taken"),
        plt.scatter([], [], marker="*", s=100, color=PALETTE["accent"], label="Outcome realized (profit/default signal)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.9,
              edgecolor=PALETTE["neutral"])

    _save(fig, "delayed_reward.png")


# ---------------------------------------------------------------------------
# 6. State dimension layer diagram
# ---------------------------------------------------------------------------

def diagram_state_layers() -> None:
    fig, ax = plt.subplots(figsize=(14, 11), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 12.0)
    ax.axis("off")

    ax.text(7.0, 11.7, "State Dimension Layers  (12D → 20D → 30D → 50D)",
            ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["text"])

    layers = [
        {
            "label":    "Layer D  (+20 features → 50D)",
            "sublabel": "Segmented history & stability descriptors",
            "features": (
                "approval_rate_new_roll_mean_4  ·  approval_rate_repeat_roll_mean_4\n"
                "approval_rate_new_lag_2  ·  approval_rate_repeat_lag_2\n"
                "expected_default_rate_new_current  ·  expected_default_rate_repeat_current\n"
                "expected_profit_new_per_accept_scaled  ·  expected_profit_repeat_per_accept_scaled\n"
                "accepted_new_share_current  ·  accepted_repeat_share_current\n"
                "reward_roll_mean_4_scaled  ·  reward_roll_std_4_scaled\n"
                "cumulative_reward_to_date_scaled  ·  cumulative_profit_to_date_scaled\n"
                "capital_usage_roll_std_4  ·  outstanding_ratio_delta_lag_1\n"
                "projected_minus_outstanding_gap  ·  threshold_gap_lag_2\n"
                "threshold_gap_delta_lag_1  ·  applications_ratio_current"
            ),
            "color":  PALETTE["layer_d"],
            "x": 0.3,  "y": 0.2,  "w": 13.0, "h": 3.4,
        },
        {
            "label":    "Layer C  (+10 features → 30D)",
            "sublabel": "Lagged memory & rolling statistics",
            "features": (
                "approval_rate_lag_2  ·  realized_default_rate_lag_2\n"
                "realized_profit_lag_2_scaled  ·  capital_usage_lag_2\n"
                "approval_rate_roll_mean_4  ·  realized_default_rate_roll_mean_4\n"
                "realized_profit_roll_mean_4_scaled  ·  realized_profit_roll_std_4_scaled\n"
                "threshold_new_delta_lag_1  ·  threshold_repeat_delta_lag_1"
            ),
            "color":  PALETTE["layer_c"],
            "x": 0.6,  "y": 3.75, "w": 12.4, "h": 2.4,
        },
        {
            "label":    "Layer B  (+8 features → 20D)",
            "sublabel": "Current economic context",
            "features": (
                "repeat_share_current  ·  expected_profit_per_application_scaled\n"
                "expected_npv_per_application_scaled  ·  realized_npv_scaled\n"
                "weekly_reward_scaled  ·  threshold_gap_normalized\n"
                "capital_headroom_ratio  ·  realized_expected_default_gap"
            ),
            "color":  PALETTE["layer_b"],
            "x": 0.9,  "y": 6.3,  "w": 11.8, "h": 2.1,
        },
        {
            "label":    "Layer A  (12D baseline)",
            "sublabel": "Core weekly portfolio snapshot — unchanged across all dimensions",
            "features": (
                "week_progress  ·  approval_rate_current\n"
                "approval_rate_new  ·  approval_rate_repeat\n"
                "rolling_realized_default_rate  ·  expected_default_rate_current\n"
                "realized_profit_scaled  ·  rolling_profit_volatility_scaled\n"
                "projected_capital_usage_ratio  ·  outstanding_ratio\n"
                "threshold_new_normalized  ·  threshold_repeat_normalized"
            ),
            "color":  PALETTE["layer_a"],
            "x": 1.2,  "y": 8.55, "w": 11.2, "h": 2.85,
        },
    ]

    for i, layer in enumerate(layers):
        # lighter layers (A, B) → dark text; darker layers (C, D) → white
        tc = PALETTE["text"] if i >= 2 else "white"

        box = FancyBboxPatch(
            (layer["x"], layer["y"]), layer["w"], layer["h"],
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=layer["color"], edgecolor=PALETTE["box_border"],
            linewidth=2.0, zorder=3 + i,
        )
        ax.add_patch(box)

        top = layer["y"] + layer["h"]
        # header label
        ax.text(layer["x"] + layer["w"] / 2, top - 0.22,
                layer["label"],
                ha="center", va="top", fontsize=11, fontweight="bold",
                color=tc, zorder=6 + i)
        # italic sublabel
        ax.text(layer["x"] + layer["w"] / 2, top - 0.5,
                layer["sublabel"],
                ha="center", va="top", fontsize=9, style="italic",
                color=tc, alpha=0.9, zorder=6 + i)
        # feature list
        ax.text(layer["x"] + layer["w"] / 2, layer["y"] + 0.15,
                layer["features"],
                ha="center", va="bottom", fontsize=8,
                color=tc, zorder=6 + i, linespacing=1.55,
                family="monospace")

    # ── size callouts on the right ─────────────────────────────────────────
    callouts = [
        (layers[3]["x"] + layers[3]["w"] + 0.15, layers[3]["y"] + layers[3]["h"] / 2, "12D"),
        (layers[2]["x"] + layers[2]["w"] + 0.15, layers[2]["y"] + layers[2]["h"] / 2, "20D"),
        (layers[1]["x"] + layers[1]["w"] + 0.15, layers[1]["y"] + layers[1]["h"] / 2, "30D"),
        (layers[0]["x"] + layers[0]["w"] + 0.15, layers[0]["y"] + layers[0]["h"] / 2, "50D"),
    ]
    for cx, cy, lbl in callouts:
        ax.text(cx, cy, lbl, ha="left", va="center", fontsize=13,
                fontweight="bold", color=PALETTE["text"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=PALETTE["box_border"], alpha=0.9))

    _save(fig, "state_dimension_layers.png")


# ---------------------------------------------------------------------------
# 7. Weekly step flowchart
# ---------------------------------------------------------------------------

def diagram_weekly_flowchart() -> None:
    # Two-column layout: steps 1-5 left, steps 6-9 right.
    # Draw order: arrows (zorder=2) -> boxes (zorder=3) -> text (zorder=4)
    # so boxes always cover arrowhead tips and text is never obscured.
    fig, ax = plt.subplots(figsize=(14, 8.0), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(-1.8, 15.5)
    ax.set_ylim(-1.6, 8.8)
    ax.axis("off")

    ax.text(6.75, 8.5, "Within-Week Simulation Step", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PALETTE["text"])

    box_w = 5.6
    box_h = 0.95
    x_L   = 0.2
    x_R   = 7.5
    gap   = 1.35

    ys_L = [7.5 - i * gap for i in range(5)]
    ys_R = [7.5 - i * gap for i in range(4)]

    # dark_bg: boxes whose background is dark enough to need white text
    dark_bg = {PALETTE["interactive"], PALETTE["terminal"]}

    actor_colors = {
        "Simulator":   "#e8f4ea",
        "Agent":       "#eaf0fb",
        "Reward fn":   "#fef6e4",
        "Obs builder": "#fef6e4",
        "Gymnasium":   "#f0eafa",
    }

    steps_L = [
        (ys_L[0], PALETTE["layer_a"],     "Applications generated",
         "New & repeat applicants arrive\n(volume proportional to weekly_apps_per_week)",
         "Simulator"),
        (ys_L[1], PALETTE["layer_b"],     "Thresholds applied",
         "Agent selects (threshold_new, threshold_repeat)\nfrom discrete grid [35...85, step 5]",
         "Agent"),
        (ys_L[2], PALETTE["layer_b"],     "Cohort accepted",
         "Applications with score above threshold accepted,\ncapital allocated, cohort stored",
         "Simulator"),
        (ys_L[3], PALETTE["layer_c"],     "Outstanding updated",
         "New cohort added to active loan book,\ncapital usage and outstanding ratio recomputed",
         "Simulator"),
        (ys_L[4], PALETTE["layer_c"],     "Past cohorts resolved",
         "Repayments, defaults and recoveries\nprocessed for cohorts past their maturity",
         "Simulator"),
    ]
    steps_R = [
        (ys_R[0], PALETTE["layer_d"],     "Reward computed",
         "profit + NPV - risk penalty - approval penalty\n- capital penalty - volatility penalty",
         "Reward fn"),
        (ys_R[1], PALETTE["layer_d"],     "Observation built",
         "12 / 20 / 30 / 50 features assembled\nfrom portfolio state and history buffers",
         "Obs builder"),
        (ys_R[2], PALETTE["interactive"], "State returned to agent",
         "Agent receives (obs, reward, done, info)\nand selects next action",
         "Gymnasium"),
        (ys_R[3], PALETTE["terminal"],    "Episode ends?",
         "If week == horizon: terminal settlement,\nall outstanding loans resolved at face value",
         "Simulator"),
    ]

    def draw_col(steps, x_box, badge_right=False):
        x_mid = x_box + box_w / 2

        # ── pass 1: connector arrows (drawn first, behind boxes) ────────────
        prev_y = None
        for y_ctr, col, title, subtitle, actor in steps:
            if prev_y is not None:
                ax.annotate(
                    "", xy=(x_mid, y_ctr + box_h / 2 + 0.08),
                    xytext=(x_mid, prev_y - box_h / 2 - 0.08),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=PALETTE["arrow"], lw=1.8),
                    zorder=2,
                )
            prev_y = y_ctr

        # ── pass 2: boxes (cover arrow tips) ────────────────────────────────
        for y_ctr, col, title, subtitle, actor in steps:
            y_bot = y_ctr - box_h / 2
            box = FancyBboxPatch(
                (x_box, y_bot), box_w, box_h,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=col, edgecolor=PALETTE["box_border"],
                linewidth=1.5, zorder=3,
            )
            ax.add_patch(box)

        # ── pass 3: text and badges (always on top) ──────────────────────────
        for y_ctr, col, title, subtitle, actor in steps:
            tc = "white" if col in dark_bg else PALETTE["text"]
            ax.text(x_mid, y_ctr + 0.19, title,
                    ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=tc, zorder=4)
            ax.text(x_mid, y_ctr - 0.21, subtitle,
                    ha="center", va="center",
                    fontsize=7.5, color=tc, zorder=4, linespacing=1.35)
            # actor badge
            ac = actor_colors.get(actor, "#f5f5f5")
            bx = (x_box + box_w + 0.12) if badge_right else (x_box - 1.25)
            badge = FancyBboxPatch(
                (bx, y_ctr - 0.2), 1.05, 0.4,
                boxstyle="round,pad=0.03",
                facecolor=ac, edgecolor=PALETTE["neutral"],
                linewidth=0.7, zorder=4,
            )
            ax.add_patch(badge)
            ax.text(bx + 0.525, y_ctr, actor,
                    ha="center", va="center",
                    fontsize=6.5, color=PALETTE["text"], zorder=5)

    draw_col(steps_L, x_L, badge_right=False)
    draw_col(steps_R, x_R, badge_right=True)

    # ── U-bend: bottom of step 5 -> floor -> right col centre -> top of step 6 ──
    x_Lc   = x_L + box_w / 2
    x_Rc   = x_R + box_w / 2
    y5_bot = ys_L[-1] - box_h / 2
    y6_top = ys_R[0]  + box_h / 2
    y_fl   = -1.2

    ax.plot([x_Lc, x_Lc], [y5_bot, y_fl], color=PALETTE["arrow"], lw=1.8, zorder=2)
    ax.plot([x_Lc, x_Rc], [y_fl,   y_fl], color=PALETTE["arrow"], lw=1.8, zorder=2)
    ax.annotate("", xy=(x_Rc, y6_top + 0.08), xytext=(x_Rc, y_fl),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["arrow"], lw=1.8),
                zorder=2)
    ax.text((x_Lc + x_Rc) / 2, y_fl - 0.18, "continue",
            ha="center", va="top", fontsize=8,
            color=PALETTE["neutral"], style="italic")

    # ── loop-back: bottom of step 9 -> floor -> left -> up -> top of step 1 ──
    y9_bot = ys_R[-1] - box_h / 2
    y1_top = ys_L[0]  + box_h / 2
    x_wall = -1.35

    ax.plot([x_Rc, x_Rc],     [y9_bot, y_fl],   color=PALETTE["accent"], lw=1.8, zorder=2)
    ax.plot([x_Rc, x_wall],   [y_fl,   y_fl],   color=PALETTE["accent"], lw=1.8, zorder=2)
    ax.plot([x_wall, x_wall], [y_fl,   y1_top], color=PALETTE["accent"], lw=1.8, zorder=2)
    ax.annotate("", xy=(x_Lc - 0.02, y1_top), xytext=(x_wall, y1_top),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["accent"], lw=1.8),
                zorder=2)
    ax.text(x_wall - 0.1, (y_fl + y1_top) / 2,
            "next\nweek", ha="right", va="center",
            fontsize=8.5, color=PALETTE["accent"], fontweight="bold")

    _save(fig, "weekly_flowchart.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating simulator diagrams …")
    diagram_episode_timeline()
    diagram_loan_lifecycle()
    diagram_delayed_reward()
    diagram_state_layers()
    diagram_weekly_flowchart()
    print("Done. All files written to outputs/diagrams/")
