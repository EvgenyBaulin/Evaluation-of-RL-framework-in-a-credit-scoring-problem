#!/usr/bin/env python3
"""Real-data comparison: default probability by within-book credit-score percentile.

Computes the empirical default rate of the Lending Club accepted-loans book by FICO
percentile, and the simulator's logistic default probability by score percentile, then
overlays them on a common 0-100 percentile axis. Also writes aggregate Lending Club
statistics used in the calibration table. Reading/parsing CSV only; no training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT = Path(__file__).resolve()
SIM_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[2]
DATA_DIR = SIM_ROOT / "data"
OUT_JSON_DIR = SIM_ROOT / "artifacts" / "real_data"
TEX_FIG_DIR = REPO_ROOT / "Tex" / "figures"

BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
USECOLS = ["loan_status", "fico_range_low", "fico_range_high", "loan_amnt", "int_rate", "term"]

# Base-market segment parameters of the simulator (see the Simulator-details appendix).
SEGMENTS = {
    "new":    {"mu": 59.0, "sd": 13.0, "beta": -2.3, "kappa": 0.18},
    "repeat": {"mu": 69.0, "sd": 10.0, "beta": -2.6, "kappa": 0.20},
}


def find_data_file() -> Path | None:
    if not DATA_DIR.exists():
        return None
    for pat in ("accepted_2007_to_2018Q4.csv", "accepted_2007_to_2018Q4.csv.gz",
                "accepted_2007_to_2018Q4.csv.zip", "accepted*.csv*"):
        hits = sorted(DATA_DIR.glob(pat))
        if hits:
            return hits[0]
    return None


def to_float_rate(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        return float(value.replace("%", "").strip())
    return float(value)


def to_term_months(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return float(digits) if digits else np.nan
    return float(value)


def pd_by_percentile(score, default, n_bins=20, min_count=500):
    """Mean default rate by score percentile band (lower score = lower percentile)."""
    pct = pd.Series(score).rank(pct=True).to_numpy() * 100.0
    edges = np.linspace(0.0, 100.0, n_bins + 1)
    idx = np.clip(np.digitize(pct, edges[1:-1]), 0, n_bins - 1)
    mids, rates = [], []
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) >= min_count:
            mids.append((edges[b] + edges[b + 1]) / 2.0)
            rates.append(float(np.mean(default[m])))
    return np.array(mids), np.array(rates)


def sim_pd_by_percentile(seg, n=400000, n_bins=20, seed=0):
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(seg["mu"], seg["sd"], n), 0.0, 100.0)
    pd_vals = 1.0 / (1.0 + np.exp(-(seg["beta"] - seg["kappa"] * ((x - 50.0) / 10.0))))
    pd_vals = np.clip(pd_vals, 0.01, 0.98)
    return pd_by_percentile(x, pd_vals, n_bins=n_bins, min_count=1)


def main() -> None:
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    TEX_FIG_DIR.mkdir(parents=True, exist_ok=True)

    data_path = find_data_file()
    if data_path is None:
        print(f"[real_data_comparison] DATA FILE NOT FOUND in {DATA_DIR}.")
        print("Place Lending Club accepted_2007_to_2018Q4.csv there and rerun.")
        print("No numbers fabricated.")
        return

    print(f"[real_data_comparison] reading {data_path.name} ...")
    comp = "gzip" if data_path.suffix == ".gz" else ("zip" if data_path.suffix == ".zip" else None)
    df = pd.read_csv(data_path, usecols=USECOLS, compression=comp, low_memory=False)

    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["y"] = df["loan_status"].isin(BAD).astype(int)
    df["fico"] = (df["fico_range_low"].astype(float) + df["fico_range_high"].astype(float)) / 2.0
    df = df.dropna(subset=["fico"])
    df["int_rate_num"] = df["int_rate"].map(to_float_rate)
    df["term_m"] = df["term"].map(to_term_months)

    lc_mids, lc_pd = pd_by_percentile(df["fico"].to_numpy(), df["y"].to_numpy())

    summary = {
        "source": "Lending Club accepted loans 2007-2018 (Kaggle wordsforthewise/lending-club)",
        "n_resolved": int(len(df)),
        "overall_default_rate": float(df["y"].mean()),
        "median_loan_amnt_usd": float(df["loan_amnt"].median()),
        "median_int_rate_pct": float(df["int_rate_num"].median()),
        "term_share_36m": float((df["term_m"] == 36).mean()),
        "term_share_60m": float((df["term_m"] == 60).mean()),
        "fico_median": float(df["fico"].median()),
        "default_rate_by_fico_percentile": [
            {"score_pct": float(m), "pd": float(p)} for m, p in zip(lc_mids, lc_pd)
        ],
    }
    (OUT_JSON_DIR / "lc_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[real_data_comparison] wrote {OUT_JSON_DIR / 'lc_summary.json'}")

    new_mids, new_pd = sim_pd_by_percentile(SEGMENTS["new"])
    rep_mids, rep_pd = sim_pd_by_percentile(SEGMENTS["repeat"])

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(lc_mids, lc_pd * 100.0, "o-", color="black", label="Lending Club (empirical)")
    plt.plot(new_mids, new_pd * 100.0, "--", color="0.35", label="Simulator, new clients")
    plt.plot(rep_mids, rep_pd * 100.0, ":", color="0.55", label="Simulator, repeat clients")
    plt.xlabel("Within-book credit-score percentile")
    plt.ylabel("Default probability (%)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(TEX_FIG_DIR / "pd_real_vs_sim.pdf")
    print(f"[real_data_comparison] wrote {TEX_FIG_DIR / 'pd_real_vs_sim.pdf'}")

    print("\n--- Lending Club aggregates (already in Table 7) ---")
    print(f"default {summary['overall_default_rate']*100:.1f}%, "
          f"median amount {summary['median_loan_amnt_usd']:.0f} USD, "
          f"median rate {summary['median_int_rate_pct']:.1f}%, "
          f"term {summary['term_share_36m']*100:.0f}/{summary['term_share_60m']*100:.0f} (36/60m)")


if __name__ == "__main__":
    main()
