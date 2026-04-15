from __future__ import annotations

from typing import Any


def compute_reward(metrics: dict[str, Any], reward_cfg: dict[str, Any], environment_cfg: dict[str, Any]) -> tuple[float, dict[str, float]]:
    delayed_reward = environment_cfg["delayed_reward"]
    profit_term = metrics["realized_profit"] if delayed_reward else metrics["expected_profit_current"]
    npv_term = metrics["realized_npv"] if delayed_reward else metrics["expected_npv_current"]

    components = {
        "profit_term": reward_cfg["profit_weight"] * profit_term,
        "npv_term": reward_cfg["npv_weight"] * npv_term,
        "risk_shaping_term": 0.0,
        "default_rate_penalty": 0.0,
        "approval_rate_penalty": 0.0,
        "capital_usage_penalty": 0.0,
        "volatility_penalty": 0.0,
    }

    if environment_cfg["risk_aware_shaping"]:
        components["risk_shaping_term"] = -reward_cfg["risk_shaping_weight"] * metrics["expected_default_rate_current"] * 100.0

    penalties = reward_cfg["penalties"]
    if penalties["default_rate"]["enabled"]:
        excess = max(0.0, metrics["rolling_realized_default_rate"] - penalties["default_rate"]["target"])
        components["default_rate_penalty"] = -penalties["default_rate"]["weight"] * (excess ** 2) * 1000.0
    if penalties["approval_rate"]["enabled"]:
        deviation = abs(metrics["approval_rate_current"] - penalties["approval_rate"]["target"])
        components["approval_rate_penalty"] = -penalties["approval_rate"]["weight"] * deviation * 1000.0
    if penalties["capital_usage"]["enabled"]:
        excess = max(0.0, metrics["projected_capital_usage_ratio"] - penalties["capital_usage"]["target"])
        components["capital_usage_penalty"] = -penalties["capital_usage"]["weight"] * (excess ** 2) * 1000.0
    if penalties["volatility"]["enabled"]:
        excess = max(0.0, metrics["rolling_profit_volatility"] - penalties["volatility"]["target"])
        scale = max(1.0, penalties["volatility"]["target"])
        components["volatility_penalty"] = -penalties["volatility"]["weight"] * (excess / scale) * 1000.0

    total_reward = float(sum(components.values()))
    return total_reward, components
