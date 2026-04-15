from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_credit_scoring_sim.env.observation_builder import ObservationBuilder
from rl_credit_scoring_sim.env.reward import compute_reward
from rl_credit_scoring_sim.env.scenarios import build_market_state


@dataclass
class PendingLoan:
    loan_id: str
    issue_week: int
    close_week: int
    segment: str
    principal: float
    score: float
    default_probability: float
    expected_profit: float
    expected_npv: float
    origin_interactive: bool


class SyntheticCreditSimulator:
    def __init__(
        self,
        config: dict[str, Any],
        scenarios: dict[str, Any],
        scale_factor: float,
        seed: int,
    ) -> None:
        self.config = config
        self.environment_cfg = config["environment"]
        self.reward_cfg = config["reward"]
        self.scenarios = scenarios
        self.scale_factor = scale_factor
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.observation_builder = ObservationBuilder(config)
        self.max_observation_dim = self.observation_builder.state_dim
        self.default_thresholds = (
            float(config["policy"]["default_threshold_new"]),
            float(config["policy"]["default_threshold_repeat"]),
        )
        self.reset_episode("base_market", seed)

    def reset_episode(self, scenario_name: str, seed: int) -> None:
        self.current_scenario_name = scenario_name
        self.scenario_cfg = self.scenarios[scenario_name]
        self.rng = np.random.default_rng(seed)
        self.absolute_week = 0
        self.interactive_week = 0
        self.pending_events: list[dict[str, Any]] = []
        self.loans: list[PendingLoan] = []
        self.loan_counter = 0
        self.week_records: list[dict[str, Any]] = []
        self.recent_profit = deque(maxlen=self.environment_cfg["reward_window"])
        self.recent_default_rate = deque(maxlen=self.environment_cfg["reward_window"])
        self.current_thresholds = self.default_thresholds
        self.last_week_metrics = self._empty_week_metrics()
        self._run_warmup()

    def _empty_week_metrics(self) -> dict[str, float]:
        return {
            "approval_rate_current": 0.0,
            "approval_rate_new": 0.0,
            "approval_rate_repeat": 0.0,
            "expected_default_rate_current": 0.0,
            "realized_profit": 0.0,
            "realized_npv": 0.0,
            "rolling_realized_default_rate": 0.0,
            "rolling_profit_volatility": 0.0,
            "projected_capital_usage_ratio": 0.0,
            "outstanding_ratio": 0.0,
            "repeat_share_current": 0.0,
            "expected_profit_current": 0.0,
            "expected_npv_current": 0.0,
            "expected_default_rate_new_current": 0.0,
            "expected_default_rate_repeat_current": 0.0,
            "expected_profit_new_current": 0.0,
            "expected_profit_repeat_current": 0.0,
            "expected_npv_new_current": 0.0,
            "expected_npv_repeat_current": 0.0,
            "applications_current": 0,
            "accepted_current": 0,
            "accepted_new_current": 0,
            "accepted_repeat_current": 0,
            "realized_default_count": 0,
            "realized_paid_count": 0,
            "realized_recovered_count": 0,
            "threshold_new": self.default_thresholds[0],
            "threshold_repeat": self.default_thresholds[1],
            "scenario_name": "base_market",
            "interactive_week": 0,
            "absolute_week": 0,
            "origin_interactive": False,
            "reward": 0.0,
            "terminal_profit": 0.0,
            "terminal_npv": 0.0,
        }

    def _run_warmup(self) -> None:
        for _ in range(self.environment_cfg["warmup_weeks"]):
            self._simulate_week(self.default_thresholds, origin_interactive=False)
            self.absolute_week += 1
        self.last_week_metrics = self.week_records[-1] if self.week_records else self._empty_week_metrics()

    def _market_state(self) -> dict[str, Any]:
        return build_market_state(
            scenario_name=self.current_scenario_name,
            scenario_cfg=self.scenario_cfg,
            week_index=self.interactive_week,
            horizon_weeks=self.environment_cfg["horizon_weeks"],
        )

    def _draw_application_counts(self, market: dict[str, Any]) -> tuple[int, int]:
        base_volume = self.environment_cfg["applications_per_week"] * self.scale_factor
        base_volume *= max(0.45, 1.0 + market["volume_trend"] + market["seasonality"])
        noise = self.rng.normal(0.0, market["volume_noise_scale"])
        base_volume *= max(0.35, 1.0 + noise)
        repeat_share = market["repeat_share"]
        repeat_count = self.rng.poisson(max(10.0, base_volume * repeat_share))
        new_count = self.rng.poisson(max(10.0, base_volume * (1.0 - repeat_share)))
        return int(new_count), int(repeat_count)

    def _generate_segment_batch(self, segment: str, count: int, params: dict[str, Any], market: dict[str, Any]) -> dict[str, np.ndarray]:
        if count <= 0:
            return {
                "scores": np.array([], dtype=float),
                "default_prob": np.array([], dtype=float),
                "will_default": np.array([], dtype=bool),
                "will_recover": np.array([], dtype=bool),
                "principal": np.array([], dtype=float),
                "duration": np.array([], dtype=int),
                "recovery_delay": np.array([], dtype=int),
                "profit_paid": np.array([], dtype=float),
            }

        scores = self.rng.normal(
            params["score_mean"],
            params["score_std"] * market["score_noise_scale"],
            size=count,
        )
        scores = np.clip(scores, 0.0, 100.0)
        default_prob = np.array([params["default_probability_fn"](score) for score in scores], dtype=float)
        default_prob = np.clip(default_prob, 0.01, 0.98)
        will_default = self.rng.random(count) < default_prob
        recovery_prob = np.full(count, params["recovery_probability"], dtype=float)
        will_recover = will_default & (self.rng.random(count) < recovery_prob)

        principal = self.rng.lognormal(
            mean=np.log(params["loan_amount_mean"]) - 0.5 * (params["loan_amount_sigma"] ** 2),
            sigma=params["loan_amount_sigma"],
            size=count,
        )
        principal = np.clip(principal * market["loan_amount_multiplier"], 300.0, 9000.0)
        duration = self.rng.integers(params["duration_min"], params["duration_max"] + 1, size=count)
        recovery_delay = self.rng.integers(1, 5, size=count)
        profit_paid = principal * (params["fee_rate"] * duration / 12.0 + params["origination_fee"])

        return {
            "scores": scores,
            "default_prob": default_prob,
            "will_default": will_default,
            "will_recover": will_recover,
            "principal": principal,
            "duration": duration,
            "recovery_delay": recovery_delay,
            "profit_paid": profit_paid,
            "late_profit_factor": np.full(count, params["late_profit_factor"], dtype=float),
        }

    def _realize_events(self, week_number: int) -> dict[str, float]:
        week_profit_all = 0.0
        week_npv_all = 0.0
        week_profit_interactive = 0.0
        week_npv_interactive = 0.0
        default_count_interactive = 0
        paid_count_interactive = 0
        recovered_count_interactive = 0

        remaining_events: list[dict[str, Any]] = []
        discount_rate_weekly = self.environment_cfg["annual_discount_rate"] / 52.0

        for event in self.pending_events:
            if event["event_week"] != week_number:
                remaining_events.append(event)
                continue
            discount = (1.0 + discount_rate_weekly) ** max(0, event["event_week"] - event["issue_week"])
            npv_value = event["cashflow"] / discount
            week_profit_all += event["cashflow"]
            week_npv_all += npv_value
            if event["origin_interactive"]:
                week_profit_interactive += event["cashflow"]
                week_npv_interactive += npv_value
                if event["kind"] == "default_loss":
                    default_count_interactive += 1
                elif event["kind"] == "paid":
                    paid_count_interactive += 1
                elif event["kind"] == "late_recovery":
                    recovered_count_interactive += 1

        self.pending_events = remaining_events
        return {
            "realized_profit_all": week_profit_all,
            "realized_npv_all": week_npv_all,
            "realized_profit_interactive": week_profit_interactive,
            "realized_npv_interactive": week_npv_interactive,
            "default_count_interactive": default_count_interactive,
            "paid_count_interactive": paid_count_interactive,
            "recovered_count_interactive": recovered_count_interactive,
        }

    def _simulate_week(self, thresholds: tuple[float, float], origin_interactive: bool) -> dict[str, Any]:
        market_state = self._market_state()
        segment_params = market_state["segment_params"]
        market = market_state["market"]
        new_count, repeat_count = self._draw_application_counts(market)
        realized = self._realize_events(self.absolute_week)

        batches = {
            "new": self._generate_segment_batch("new", new_count, segment_params["new"], market),
            "repeat": self._generate_segment_batch("repeat", repeat_count, segment_params["repeat"], market),
        }
        thresholds_map = {"new": thresholds[0], "repeat": thresholds[1]}

        total_apps = new_count + repeat_count
        total_accepted = 0
        accepted_defaults_probability_sum = 0.0
        expected_profit_current = 0.0
        expected_npv_current = 0.0
        total_principal_accepted = 0.0
        segment_accepts = {"new": 0, "repeat": 0}
        segment_expected_profit = {"new": 0.0, "repeat": 0.0}
        segment_expected_npv = {"new": 0.0, "repeat": 0.0}
        segment_default_probability_sum = {"new": 0.0, "repeat": 0.0}

        discount_rate_weekly = self.environment_cfg["annual_discount_rate"] / 52.0

        for segment, batch in batches.items():
            accepted_mask = batch["scores"] >= thresholds_map[segment]
            segment_accepts[segment] = int(accepted_mask.sum())
            total_accepted += int(accepted_mask.sum())
            if not accepted_mask.any():
                continue

            scores = batch["scores"][accepted_mask]
            default_prob = batch["default_prob"][accepted_mask]
            will_default = batch["will_default"][accepted_mask]
            will_recover = batch["will_recover"][accepted_mask]
            principal = batch["principal"][accepted_mask]
            duration = batch["duration"][accepted_mask]
            recovery_delay = batch["recovery_delay"][accepted_mask]
            profit_paid = batch["profit_paid"][accepted_mask]
            late_profit_factor = batch["late_profit_factor"][accepted_mask]

            expected_recovery_value = principal + profit_paid * late_profit_factor
            expected_profit = (
                (1.0 - default_prob) * profit_paid
                + default_prob * segment_params[segment]["recovery_probability"] * (profit_paid * late_profit_factor)
                - default_prob * (1.0 - segment_params[segment]["recovery_probability"]) * principal
            )
            expected_profit_current += float(expected_profit.sum())
            segment_expected_profit[segment] += float(expected_profit.sum())
            expected_npv_current += float(
                (
                    expected_profit
                    / np.power(1.0 + discount_rate_weekly, duration + default_prob * recovery_delay)
                ).sum()
            )
            segment_expected_npv[segment] += float(
                (
                    expected_profit
                    / np.power(1.0 + discount_rate_weekly, duration + default_prob * recovery_delay)
                ).sum()
            )
            accepted_defaults_probability_sum += float(default_prob.sum())
            segment_default_probability_sum[segment] += float(default_prob.sum())
            total_principal_accepted += float(principal.sum())

            for idx in range(scores.shape[0]):
                self.loan_counter += 1
                close_week = self.absolute_week + int(duration[idx] + (recovery_delay[idx] if will_recover[idx] else 0))
                loan = PendingLoan(
                    loan_id=f"loan_{self.loan_counter}",
                    issue_week=self.absolute_week,
                    close_week=close_week,
                    segment=segment,
                    principal=float(principal[idx]),
                    score=float(scores[idx]),
                    default_probability=float(default_prob[idx]),
                    expected_profit=float(expected_profit[idx]),
                    expected_npv=float(
                        expected_profit[idx]
                        / ((1.0 + discount_rate_weekly) ** max(1, int(duration[idx])))
                    ),
                    origin_interactive=origin_interactive,
                )
                self.loans.append(loan)
                due_week = self.absolute_week + int(duration[idx])
                if not will_default[idx]:
                    self.pending_events.append(
                        {
                            "event_week": due_week,
                            "issue_week": self.absolute_week,
                            "cashflow": float(profit_paid[idx]),
                            "kind": "paid",
                            "origin_interactive": origin_interactive,
                        }
                    )
                else:
                    self.pending_events.append(
                        {
                            "event_week": due_week,
                            "issue_week": self.absolute_week,
                            "cashflow": float(-principal[idx]),
                            "kind": "default_loss",
                            "origin_interactive": origin_interactive,
                        }
                    )
                    if will_recover[idx]:
                        self.pending_events.append(
                            {
                                "event_week": due_week + int(recovery_delay[idx]),
                                "issue_week": self.absolute_week,
                                "cashflow": float(expected_recovery_value[idx]),
                                "kind": "late_recovery",
                                "origin_interactive": origin_interactive,
                            }
                        )

        outstanding_principal = sum(
            loan.principal for loan in self.loans if loan.issue_week <= self.absolute_week < loan.close_week
        )
        projected_capital_usage_ratio = (outstanding_principal + total_principal_accepted) / self.environment_cfg["capital_limit"]
        realized_default_rate = (
            realized["default_count_interactive"]
            / max(
                1,
                realized["default_count_interactive"] + realized["paid_count_interactive"] + realized["recovered_count_interactive"],
            )
        )
        self.recent_default_rate.append(realized_default_rate)
        self.recent_profit.append(realized["realized_profit_interactive"])

        metrics = {
            "approval_rate_current": total_accepted / max(total_apps, 1),
            "approval_rate_new": segment_accepts["new"] / max(new_count, 1),
            "approval_rate_repeat": segment_accepts["repeat"] / max(repeat_count, 1),
            "expected_default_rate_current": accepted_defaults_probability_sum / max(total_accepted, 1),
            "realized_profit": realized["realized_profit_interactive"],
            "realized_npv": realized["realized_npv_interactive"],
            "rolling_realized_default_rate": float(np.mean(self.recent_default_rate)) if self.recent_default_rate else 0.0,
            "rolling_profit_volatility": float(np.std(self.recent_profit)) if len(self.recent_profit) > 1 else 0.0,
            "projected_capital_usage_ratio": projected_capital_usage_ratio,
            "outstanding_ratio": outstanding_principal / self.environment_cfg["capital_limit"],
            "repeat_share_current": repeat_count / max(total_apps, 1),
            "expected_profit_current": expected_profit_current,
            "expected_npv_current": expected_npv_current,
            "expected_default_rate_new_current": segment_default_probability_sum["new"] / max(segment_accepts["new"], 1),
            "expected_default_rate_repeat_current": segment_default_probability_sum["repeat"] / max(
                segment_accepts["repeat"], 1
            ),
            "expected_profit_new_current": segment_expected_profit["new"],
            "expected_profit_repeat_current": segment_expected_profit["repeat"],
            "expected_npv_new_current": segment_expected_npv["new"],
            "expected_npv_repeat_current": segment_expected_npv["repeat"],
            "applications_current": total_apps,
            "accepted_current": total_accepted,
            "accepted_new_current": segment_accepts["new"],
            "accepted_repeat_current": segment_accepts["repeat"],
            "realized_default_count": realized["default_count_interactive"],
            "realized_paid_count": realized["paid_count_interactive"],
            "realized_recovered_count": realized["recovered_count_interactive"],
            "threshold_new": thresholds[0],
            "threshold_repeat": thresholds[1],
            "scenario_name": self.current_scenario_name,
            "interactive_week": self.interactive_week,
            "absolute_week": self.absolute_week,
            "origin_interactive": origin_interactive,
        }
        reward, reward_components = compute_reward(metrics, self.reward_cfg, self.environment_cfg)
        metrics["reward"] = reward
        metrics.update(reward_components)
        self.week_records.append(metrics)
        return metrics

    def _terminal_settlement(self) -> dict[str, float]:
        discount_rate_weekly = self.environment_cfg["annual_discount_rate"] / 52.0
        settlement_profit = 0.0
        settlement_npv = 0.0
        terminal_default_count = 0
        terminal_paid_count = 0
        terminal_recovered_count = 0
        remaining_events = []
        for event in self.pending_events:
            if not event["origin_interactive"]:
                remaining_events.append(event)
                continue
            distance = max(0, event["event_week"] - self.absolute_week)
            settlement_profit += event["cashflow"]
            settlement_npv += event["cashflow"] / ((1.0 + discount_rate_weekly) ** distance)
            if event["kind"] == "default_loss":
                terminal_default_count += 1
            elif event["kind"] == "paid":
                terminal_paid_count += 1
            elif event["kind"] == "late_recovery":
                terminal_recovered_count += 1
        self.pending_events = remaining_events
        return {
            "terminal_profit": settlement_profit,
            "terminal_npv": settlement_npv,
            "terminal_default_count": terminal_default_count,
            "terminal_paid_count": terminal_paid_count,
            "terminal_recovered_count": terminal_recovered_count,
        }

    def step(self, thresholds: tuple[float, float]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.current_thresholds = thresholds
        metrics = self._simulate_week(thresholds, origin_interactive=True)
        self.interactive_week += 1
        self.absolute_week += 1
        done = self.interactive_week >= self.environment_cfg["horizon_weeks"]
        if done and self.environment_cfg["terminal_settlement"]:
            settlement = self._terminal_settlement()
            metrics["reward"] += settlement["terminal_profit"] + self.reward_cfg["npv_weight"] * settlement["terminal_npv"]
            metrics["realized_profit"] += settlement["terminal_profit"]
            metrics["realized_npv"] += settlement["terminal_npv"]
            metrics["terminal_profit"] = settlement["terminal_profit"]
            metrics["terminal_npv"] = settlement["terminal_npv"]
            metrics["realized_default_count"] += settlement["terminal_default_count"]
            metrics["realized_paid_count"] += settlement["terminal_paid_count"]
            metrics["realized_recovered_count"] += settlement["terminal_recovered_count"]
        else:
            metrics["terminal_profit"] = 0.0
            metrics["terminal_npv"] = 0.0
        self.last_week_metrics = metrics
        return self.get_observation(), float(metrics["reward"]), done, {"week_metrics": metrics}

    def get_observation(self) -> np.ndarray:
        return self.observation_builder.build(self)

    def get_reset_info(self) -> dict[str, Any]:
        return {
            "scenario_name": self.current_scenario_name,
            "week_metrics": self.last_week_metrics,
            "current_thresholds": self.current_thresholds,
        }

    def score_action_candidates(self, candidate_thresholds: tuple[float, float], preview_scale: float = 1.0) -> dict[str, float]:
        market_state = self._market_state()
        segment_params = market_state["segment_params"]
        market = market_state["market"]
        preview_count = int(self.environment_cfg["applications_per_week"] * self.scale_factor * preview_scale)
        repeat_share = market["repeat_share"]
        new_count = max(8, int(preview_count * (1.0 - repeat_share)))
        repeat_count = max(8, int(preview_count * repeat_share))
        batches = {
            "new": self._generate_segment_batch("new", new_count, segment_params["new"], market),
            "repeat": self._generate_segment_batch("repeat", repeat_count, segment_params["repeat"], market),
        }
        approvals = []
        expected_defaults = []
        expected_profit = 0.0
        expected_npv = 0.0
        principal_accepted = 0.0
        discount_rate_weekly = self.environment_cfg["annual_discount_rate"] / 52.0
        thresholds_map = {"new": candidate_thresholds[0], "repeat": candidate_thresholds[1]}
        for segment, batch in batches.items():
            accepted_mask = batch["scores"] >= thresholds_map[segment]
            if accepted_mask.any():
                default_prob = batch["default_prob"][accepted_mask]
                principal = batch["principal"][accepted_mask]
                duration = batch["duration"][accepted_mask]
                profit_paid = batch["profit_paid"][accepted_mask]
                late_factor = batch["late_profit_factor"][accepted_mask]
                recovery_prob = segment_params[segment]["recovery_probability"]
                expected_segment_profit = (
                    (1.0 - default_prob) * profit_paid
                    + default_prob * recovery_prob * (profit_paid * late_factor)
                    - default_prob * (1.0 - recovery_prob) * principal
                )
                expected_profit += float(expected_segment_profit.sum())
                expected_npv += float(
                    (expected_segment_profit / np.power(1.0 + discount_rate_weekly, duration)).sum()
                )
                expected_defaults.append(float(default_prob.mean()))
                approvals.append(float(accepted_mask.mean()))
                principal_accepted += float(principal.sum())
            else:
                expected_defaults.append(0.0)
                approvals.append(0.0)
        capital_ratio = (
            sum(loan.principal for loan in self.loans if loan.issue_week <= self.absolute_week < loan.close_week) + principal_accepted
        ) / self.environment_cfg["capital_limit"]
        approval_rate = (
            approvals[0] * new_count + approvals[1] * repeat_count
        ) / max(1, new_count + repeat_count)
        expected_default_rate = float(np.mean(expected_defaults))
        preview_metrics = {
            "approval_rate_current": approval_rate,
            "approval_rate_new": approvals[0],
            "approval_rate_repeat": approvals[1],
            "expected_default_rate_current": expected_default_rate,
            "realized_profit": self.last_week_metrics.get("realized_profit", 0.0),
            "realized_npv": self.last_week_metrics.get("realized_npv", 0.0),
            "rolling_realized_default_rate": self.last_week_metrics.get("rolling_realized_default_rate", 0.0),
            "rolling_profit_volatility": self.last_week_metrics.get("rolling_profit_volatility", 0.0),
            "projected_capital_usage_ratio": capital_ratio,
            "outstanding_ratio": capital_ratio,
            "repeat_share_current": repeat_count / max(1, new_count + repeat_count),
            "expected_profit_current": expected_profit,
            "expected_npv_current": expected_npv,
        }
        objective, _ = compute_reward(preview_metrics, self.reward_cfg, self.environment_cfg)
        preview_metrics["objective"] = objective
        return preview_metrics

    def episode_dataframe(self, controller_name: str, seed: int, run_id: int) -> list[dict[str, Any]]:
        records = []
        for record in self.week_records:
            if not record["origin_interactive"]:
                continue
            row = dict(record)
            row["controller"] = controller_name
            row["seed"] = seed
            row["run_id"] = run_id
            records.append(row)
        return records

    def episode_summary(self) -> dict[str, float]:
        interactive_loans = [loan for loan in self.loans if loan.origin_interactive]
        if not interactive_loans:
            return {
                "approval_rate": 0.0,
                "default_rate": 0.0,
                "expected_profit": 0.0,
                "npv": 0.0,
                "cumulative_reward": 0.0,
                "capital_usage_mean": 0.0,
                "reward_volatility": 0.0,
                "stability_index": 0.0,
                "threshold_volatility": 0.0,
            }
        history = [record for record in self.week_records if record["origin_interactive"]]
        accepted_total = sum(record["accepted_current"] for record in history)
        application_total = sum(record["applications_current"] for record in history)
        default_total = sum(record["realized_default_count"] for record in history)
        threshold_pairs = np.array([[record["threshold_new"], record["threshold_repeat"]] for record in history], dtype=float)
        threshold_changes = np.diff(threshold_pairs, axis=0) if len(threshold_pairs) > 1 else np.zeros((1, 2))
        rewards = np.array([record["reward"] for record in history], dtype=float)
        profits = np.array([record["realized_profit"] for record in history], dtype=float)
        return {
            "approval_rate": accepted_total / max(1, application_total),
            "default_rate": default_total / max(1, accepted_total),
            "expected_profit": float(profits.sum()),
            "npv": float(sum(record["realized_npv"] for record in history)),
            "cumulative_reward": float(rewards.sum()),
            "capital_usage_mean": float(np.mean([record["projected_capital_usage_ratio"] for record in history])),
            "reward_volatility": float(np.std(rewards)),
            "stability_index": float(1.0 / (1.0 + np.std(rewards))),
            "threshold_volatility": float(np.std(threshold_changes)),
        }
