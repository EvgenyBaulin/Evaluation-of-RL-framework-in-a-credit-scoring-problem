from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BaselineDecision:
    thresholds: tuple[float, float]
    action: Any


class BaseBaseline:
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.policy_config = config["policy"]
        self.default_shared = (
            float(config["policy"]["default_threshold_new"]),
            float(config["policy"]["default_threshold_new"]),
        )
        self.default_split = (
            float(config["policy"]["default_threshold_new"]),
            float(config["policy"]["default_threshold_repeat"]),
        )
        self.current_thresholds = self.default_split

    def begin_episode(self) -> None:
        self.current_thresholds = self.default_split

    def predict(self, observation: np.ndarray, env) -> BaselineDecision:
        raise NotImplementedError


class StaticThresholdBaseline(BaseBaseline):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("static_threshold", config)
        shared_threshold = float(
            (config["policy"]["default_threshold_new"] + config["policy"]["default_threshold_repeat"]) / 2.0
        )
        self.static_thresholds = (shared_threshold, shared_threshold)

    def predict(self, observation: np.ndarray, env) -> BaselineDecision:
        action = env.thresholds_to_action(self.static_thresholds)
        return BaselineDecision(thresholds=self.static_thresholds, action=action)


class SplitPolicyStaticBaseline(BaseBaseline):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("split_policy_static", config)

    def predict(self, observation: np.ndarray, env) -> BaselineDecision:
        action = env.thresholds_to_action(self.default_split)
        return BaselineDecision(thresholds=self.default_split, action=action)


class ProfitOrientedBaseline(BaseBaseline):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("profit_oriented", config)

    def predict(self, observation: np.ndarray, env) -> BaselineDecision:
        best_thresholds = self.default_split
        best_score = -np.inf
        for thresholds in env.action_map:
            preview = env.preview_action(thresholds)
            score = preview["expected_profit_current"]
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
        self.current_thresholds = best_thresholds
        return BaselineDecision(thresholds=best_thresholds, action=env.thresholds_to_action(best_thresholds))


class RiskAwareWeeklyBaseline(BaseBaseline):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("risk_aware_weekly", config)
        self.step = float(config["policy"]["threshold_granularity"])

    def predict(self, observation: np.ndarray, env) -> BaselineDecision:
        metrics = env.get_week_metrics()
        new_threshold, repeat_threshold = self.current_thresholds

        if metrics["rolling_realized_default_rate"] > 0.12:
            new_threshold += self.step
            repeat_threshold += 0.5 * self.step
        elif metrics["approval_rate_current"] < 0.32:
            new_threshold -= 0.5 * self.step
            repeat_threshold -= self.step

        if metrics["projected_capital_usage_ratio"] > 0.82:
            new_threshold += self.step
            repeat_threshold += self.step

        if metrics["approval_rate_new"] > metrics["approval_rate_repeat"] + 0.1:
            new_threshold += self.step
        if metrics["approval_rate_repeat"] < 0.25:
            repeat_threshold -= 0.5 * self.step

        threshold_min = self.policy_config["threshold_min"]
        threshold_max = self.policy_config["threshold_max"]
        new_threshold = float(np.clip(np.round(new_threshold / self.step) * self.step, threshold_min, threshold_max))
        repeat_threshold = float(
            np.clip(np.round(repeat_threshold / self.step) * self.step, threshold_min, threshold_max)
        )
        self.current_thresholds = (new_threshold, repeat_threshold)
        return BaselineDecision(
            thresholds=self.current_thresholds,
            action=env.thresholds_to_action(self.current_thresholds),
        )


class ConstraintAwareWeeklyBaseline(BaseBaseline):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("constraint_aware_weekly", config)
        penalties = config["reward"]["penalties"]
        self.targets = {
            "default": penalties["default_rate"]["target"],
            "approval": penalties["approval_rate"]["target"],
            "capital": penalties["capital_usage"]["target"],
        }

    def predict(self, observation: np.ndarray, env) -> BaselineDecision:
        best_thresholds = self.current_thresholds
        best_score = -np.inf
        for thresholds in env.action_map:
            preview = env.preview_action(thresholds)
            violation = 0.0
            violation += max(0.0, preview["expected_default_rate_current"] - self.targets["default"]) * 3000.0
            violation += abs(preview["approval_rate_current"] - self.targets["approval"]) * 1500.0
            violation += max(0.0, preview["projected_capital_usage_ratio"] - self.targets["capital"]) * 2500.0
            score = preview["expected_profit_current"] + 0.3 * preview["expected_npv_current"] - violation
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
        self.current_thresholds = best_thresholds
        return BaselineDecision(thresholds=best_thresholds, action=env.thresholds_to_action(best_thresholds))
