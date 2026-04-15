from __future__ import annotations

import math
from copy import deepcopy


BASE_SEGMENT_PARAMS = {
    "new": {
        "score_mean": 59.0,
        "score_std": 13.0,
        "default_intercept": -1.15,
        "score_sensitivity": 0.18,
        "loan_amount_mean": 1650.0,
        "loan_amount_sigma": 0.28,
        "duration_min": 5,
        "duration_max": 12,
        "fee_rate": 0.16,
        "origination_fee": 0.035,
        "recovery_probability": 0.26,
        "late_profit_factor": 0.65,
    },
    "repeat": {
        "score_mean": 69.0,
        "score_std": 10.0,
        "default_intercept": -1.75,
        "score_sensitivity": 0.20,
        "loan_amount_mean": 2150.0,
        "loan_amount_sigma": 0.25,
        "duration_min": 4,
        "duration_max": 10,
        "fee_rate": 0.13,
        "origination_fee": 0.03,
        "recovery_probability": 0.34,
        "late_profit_factor": 0.72,
    },
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def interpolate_weekly_shift(total_shift: float, week_index: int, horizon_weeks: int) -> float:
    if horizon_weeks <= 1:
        return total_shift
    progress = week_index / max(horizon_weeks - 1, 1)
    return total_shift * progress


def build_market_state(
    scenario_name: str,
    scenario_cfg: dict,
    week_index: int,
    horizon_weeks: int,
) -> dict:
    state = deepcopy(BASE_SEGMENT_PARAMS)
    shock_multiplier = 0.0
    shock_start = scenario_cfg.get("shock_start_week")
    if shock_start is not None and week_index >= shock_start:
        shock_multiplier = scenario_cfg.get("shock_magnitude", 0.0)

    repeat_share = scenario_cfg["repeat_share_base"] + 0.08 * math.sin(
        2 * math.pi * (week_index / max(horizon_weeks, 1))
    )
    repeat_share += 0.04 * interpolate_weekly_shift(scenario_cfg.get("weekly_volume_trend", 0.0), week_index, horizon_weeks)
    repeat_share = max(0.15, min(0.75, repeat_share))

    market = {
        "scenario_name": scenario_name,
        "repeat_share": repeat_share,
        "volume_trend": interpolate_weekly_shift(scenario_cfg.get("weekly_volume_trend", 0.0), week_index, horizon_weeks),
        "seasonality": scenario_cfg.get("seasonal_amplitude", 0.0) * math.sin(2 * math.pi * week_index / 13.0),
        "score_noise_scale": scenario_cfg.get("score_noise_scale", 1.0),
        "volume_noise_scale": scenario_cfg.get("volume_noise_scale", 0.08),
        "loan_amount_multiplier": scenario_cfg.get("loan_amount_multiplier", 1.0),
        "shock_multiplier": shock_multiplier,
    }

    state["new"]["score_mean"] += 10.0 * interpolate_weekly_shift(
        scenario_cfg.get("score_shift_new", 0.0), week_index, horizon_weeks
    )
    state["repeat"]["score_mean"] += 10.0 * interpolate_weekly_shift(
        scenario_cfg.get("score_shift_repeat", 0.0), week_index, horizon_weeks
    )

    state["new"]["default_intercept"] += interpolate_weekly_shift(
        scenario_cfg.get("default_shift_new", 0.0), week_index, horizon_weeks
    ) + shock_multiplier
    state["repeat"]["default_intercept"] += interpolate_weekly_shift(
        scenario_cfg.get("default_shift_repeat", 0.0), week_index, horizon_weeks
    ) + 0.75 * shock_multiplier

    state["new"]["recovery_probability"] = max(
        0.05,
        min(0.85, state["new"]["recovery_probability"] + scenario_cfg.get("recovery_shift", 0.0)),
    )
    state["repeat"]["recovery_probability"] = max(
        0.05,
        min(0.90, state["repeat"]["recovery_probability"] + scenario_cfg.get("recovery_shift", 0.0)),
    )

    for segment in ("new", "repeat"):
        state[segment]["default_probability_fn"] = lambda score, params=state[segment]: _sigmoid(
            params["default_intercept"] - params["score_sensitivity"] * ((score - 50.0) / 10.0)
        )
    return {"segment_params": state, "market": market}
