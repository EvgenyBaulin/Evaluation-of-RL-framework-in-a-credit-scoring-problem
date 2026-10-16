from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ALLOWED_STATE_DIMS = (12, 20, 30, 50)


@dataclass(frozen=True)
class ObservationFeature:
    name: str
    layer: str
    definition: str
    feature_type: str
    normalized: bool
    extractor: Callable[[Any], float]


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-9:
        return 0.0
    return float(numerator) / float(denominator)


def _last_metrics(simulator) -> dict[str, Any]:
    return simulator.last_week_metrics


def _interactive_history(simulator) -> list[dict[str, Any]]:
    return [record for record in simulator.week_records if record["origin_interactive"]]


def _interactive_record(simulator, lag: int) -> dict[str, Any] | None:
    history = _interactive_history(simulator)
    if len(history) < lag:
        return None
    return history[-lag]


def _metric(simulator, key: str, default: float = 0.0) -> float:
    value = _last_metrics(simulator).get(key, default)
    return float(value if value is not None else default)


def _record_metric(simulator, lag: int, key: str, default: float = 0.0) -> float:
    record = _interactive_record(simulator, lag)
    if record is None:
        return default
    value = record.get(key, default)
    return float(value if value is not None else default)


def _threshold_span(simulator) -> float:
    policy_cfg = simulator.config["policy"]
    return max(1.0, float(policy_cfg["threshold_max"] - policy_cfg["threshold_min"]))


def _application_scale(simulator) -> float:
    base = float(simulator.environment_cfg["applications_per_week"]) * float(simulator.scale_factor)
    return max(1.0, base)


def _realized_default_rate(record: dict[str, Any] | None) -> float:
    if record is None:
        return 0.0
    resolved = (
        float(record.get("realized_default_count", 0.0))
        + float(record.get("realized_paid_count", 0.0))
        + float(record.get("realized_recovered_count", 0.0))
    )
    return _safe_div(record.get("realized_default_count", 0.0), resolved)


def _rolling_reduce(
    simulator,
    extractor: Callable[[dict[str, Any]], float],
    window: int,
    reducer: Callable[[np.ndarray], float],
) -> float:
    history = _interactive_history(simulator)
    if not history:
        return 0.0
    values = np.asarray([extractor(record) for record in history[-window:]], dtype=float)
    if values.size == 0:
        return 0.0
    if values.size == 1 and reducer is np.std:
        return 0.0
    return float(reducer(values))


def _threshold_gap_from_record(record: dict[str, Any] | None, simulator) -> float:
    if record is None:
        return 0.0
    return _safe_div(record.get("threshold_new", 0.0) - record.get("threshold_repeat", 0.0), _threshold_span(simulator))


def _threshold_delta(simulator, key: str) -> float:
    current = _interactive_record(simulator, 1)
    previous = _interactive_record(simulator, 2)
    if current is None or previous is None:
        return 0.0
    return _safe_div(current.get(key, 0.0) - previous.get(key, 0.0), _threshold_span(simulator))


def _accepted_share(simulator, key: str) -> float:
    accepted_total = _metric(simulator, "accepted_current")
    return _safe_div(_metric(simulator, key), accepted_total)


def _per_application(simulator, key: str, scale: float = 100.0) -> float:
    applications = _metric(simulator, "applications_current")
    return _safe_div(_metric(simulator, key), max(1.0, applications) * scale)


def _per_accept(simulator, numerator_key: str, accepted_key: str, scale: float = 100.0) -> float:
    accepted = _metric(simulator, accepted_key)
    return _safe_div(_metric(simulator, numerator_key), max(1.0, accepted) * scale)


def _cumulative_metric(simulator, key: str, scale: float) -> float:
    history = _interactive_history(simulator)
    return float(sum(float(record.get(key, 0.0)) for record in history) / scale)


OBSERVATION_FEATURES: list[ObservationFeature] = [
    ObservationFeature(
        name="week_progress",
        layer="A",
        definition="Interactive week index divided by the episode horizon.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _safe_div(simulator.interactive_week, max(1, simulator.environment_cfg["horizon_weeks"])),
    ),
    ObservationFeature(
        name="approval_rate_current",
        layer="A",
        definition="Last observed overall approval rate.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "approval_rate_current"),
    ),
    ObservationFeature(
        name="approval_rate_new",
        layer="A",
        definition="Last observed approval rate for new applicants.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "approval_rate_new"),
    ),
    ObservationFeature(
        name="approval_rate_repeat",
        layer="A",
        definition="Last observed approval rate for repeat applicants.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "approval_rate_repeat"),
    ),
    ObservationFeature(
        name="rolling_realized_default_rate",
        layer="A",
        definition="Rolling realized default rate from the simulator reward window.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "rolling_realized_default_rate"),
    ),
    ObservationFeature(
        name="expected_default_rate_current",
        layer="A",
        definition="Last observed expected default rate among accepted applications.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "expected_default_rate_current"),
    ),
    ObservationFeature(
        name="realized_profit_scaled",
        layer="A",
        definition="Last observed realized profit scaled by 10,000.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "realized_profit") / 10000.0,
    ),
    ObservationFeature(
        name="rolling_profit_volatility_scaled",
        layer="A",
        definition="Rolling realized profit volatility scaled by 10,000.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "rolling_profit_volatility") / 10000.0,
    ),
    ObservationFeature(
        name="projected_capital_usage_ratio",
        layer="A",
        definition="Projected capital usage ratio after the last action.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "projected_capital_usage_ratio"),
    ),
    ObservationFeature(
        name="outstanding_ratio",
        layer="A",
        definition="Outstanding principal divided by the capital limit.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "outstanding_ratio"),
    ),
    ObservationFeature(
        name="threshold_new_normalized",
        layer="A",
        definition="Last applied new-client threshold normalized to the configured threshold range.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _safe_div(
            _metric(simulator, "threshold_new", simulator.current_thresholds[0]) - simulator.config["policy"]["threshold_min"],
            _threshold_span(simulator),
        ),
    ),
    ObservationFeature(
        name="threshold_repeat_normalized",
        layer="A",
        definition="Last applied repeat-client threshold normalized to the configured threshold range.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _safe_div(
            _metric(simulator, "threshold_repeat", simulator.current_thresholds[1]) - simulator.config["policy"]["threshold_min"],
            _threshold_span(simulator),
        ),
    ),
    ObservationFeature(
        name="repeat_share_current",
        layer="B",
        definition="Share of applications coming from repeat clients in the last observed week.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "repeat_share_current"),
    ),
    ObservationFeature(
        name="expected_profit_per_application_scaled",
        layer="B",
        definition="Last observed expected cohort profit per application scaled by 100.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _per_application(simulator, "expected_profit_current", scale=100.0),
    ),
    ObservationFeature(
        name="expected_npv_per_application_scaled",
        layer="B",
        definition="Last observed expected cohort NPV per application scaled by 100.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _per_application(simulator, "expected_npv_current", scale=100.0),
    ),
    ObservationFeature(
        name="realized_npv_scaled",
        layer="B",
        definition="Last observed realized NPV scaled by 10,000.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "realized_npv") / 10000.0,
    ),
    ObservationFeature(
        name="weekly_reward_scaled",
        layer="B",
        definition="Last observed weekly reward scaled by 10,000.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "reward") / 10000.0,
    ),
    ObservationFeature(
        name="threshold_gap_normalized",
        layer="B",
        definition="Difference between new and repeat thresholds normalized by the threshold range.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _safe_div(
            _metric(simulator, "threshold_new", simulator.current_thresholds[0])
            - _metric(simulator, "threshold_repeat", simulator.current_thresholds[1]),
            _threshold_span(simulator),
        ),
    ),
    ObservationFeature(
        name="capital_headroom_ratio",
        layer="B",
        definition="Remaining capital headroom after the last action.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: 1.0 - _metric(simulator, "projected_capital_usage_ratio"),
    ),
    ObservationFeature(
        name="realized_expected_default_gap",
        layer="B",
        definition="Gap between rolling realized default rate and last observed expected default rate.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "rolling_realized_default_rate")
        - _metric(simulator, "expected_default_rate_current"),
    ),
    ObservationFeature(
        name="approval_rate_lag_2",
        layer="C",
        definition="Overall approval rate observed two interactive weeks ago.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _record_metric(simulator, 2, "approval_rate_current"),
    ),
    ObservationFeature(
        name="realized_default_rate_lag_2",
        layer="C",
        definition="Realized default rate observed two interactive weeks ago.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _realized_default_rate(_interactive_record(simulator, 2)),
    ),
    ObservationFeature(
        name="realized_profit_lag_2_scaled",
        layer="C",
        definition="Realized profit from two interactive weeks ago scaled by 10,000.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _record_metric(simulator, 2, "realized_profit") / 10000.0,
    ),
    ObservationFeature(
        name="capital_usage_lag_2",
        layer="C",
        definition="Projected capital usage ratio observed two interactive weeks ago.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _record_metric(simulator, 2, "projected_capital_usage_ratio"),
    ),
    ObservationFeature(
        name="approval_rate_roll_mean_4",
        layer="C",
        definition="Mean approval rate over the last four interactive weeks.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("approval_rate_current", 0.0)),
            window=4,
            reducer=np.mean,
        ),
    ),
    ObservationFeature(
        name="realized_default_rate_roll_mean_4",
        layer="C",
        definition="Mean realized default rate over the last four interactive weeks.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: _realized_default_rate(record),
            window=4,
            reducer=np.mean,
        ),
    ),
    ObservationFeature(
        name="realized_profit_roll_mean_4_scaled",
        layer="C",
        definition="Mean realized profit over the last four interactive weeks scaled by 10,000.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("realized_profit", 0.0)) / 10000.0,
            window=4,
            reducer=np.mean,
        ),
    ),
    ObservationFeature(
        name="realized_profit_roll_std_4_scaled",
        layer="C",
        definition="Standard deviation of realized profit over the last four interactive weeks scaled by 10,000.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("realized_profit", 0.0)) / 10000.0,
            window=4,
            reducer=np.std,
        ),
    ),
    ObservationFeature(
        name="threshold_new_delta_lag_1",
        layer="C",
        definition="Week-over-week change in the new-client threshold normalized by the threshold range.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _threshold_delta(simulator, "threshold_new"),
    ),
    ObservationFeature(
        name="threshold_repeat_delta_lag_1",
        layer="C",
        definition="Week-over-week change in the repeat-client threshold normalized by the threshold range.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _threshold_delta(simulator, "threshold_repeat"),
    ),
    ObservationFeature(
        name="approval_rate_new_roll_mean_4",
        layer="D",
        definition="Mean new-client approval rate over the last four interactive weeks.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("approval_rate_new", 0.0)),
            window=4,
            reducer=np.mean,
        ),
    ),
    ObservationFeature(
        name="approval_rate_repeat_roll_mean_4",
        layer="D",
        definition="Mean repeat-client approval rate over the last four interactive weeks.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("approval_rate_repeat", 0.0)),
            window=4,
            reducer=np.mean,
        ),
    ),
    ObservationFeature(
        name="approval_rate_new_lag_2",
        layer="D",
        definition="New-client approval rate observed two interactive weeks ago.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _record_metric(simulator, 2, "approval_rate_new"),
    ),
    ObservationFeature(
        name="approval_rate_repeat_lag_2",
        layer="D",
        definition="Repeat-client approval rate observed two interactive weeks ago.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _record_metric(simulator, 2, "approval_rate_repeat"),
    ),
    ObservationFeature(
        name="expected_default_rate_new_current",
        layer="D",
        definition="Last observed expected default rate for accepted new-client applications.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "expected_default_rate_new_current"),
    ),
    ObservationFeature(
        name="expected_default_rate_repeat_current",
        layer="D",
        definition="Last observed expected default rate for accepted repeat-client applications.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "expected_default_rate_repeat_current"),
    ),
    ObservationFeature(
        name="expected_profit_new_per_accept_scaled",
        layer="D",
        definition="Expected profit of accepted new-client loans per accepted loan, scaled by 100.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _per_accept(simulator, "expected_profit_new_current", "accepted_new_current", scale=100.0),
    ),
    ObservationFeature(
        name="expected_profit_repeat_per_accept_scaled",
        layer="D",
        definition="Expected profit of accepted repeat-client loans per accepted loan, scaled by 100.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _per_accept(
            simulator,
            "expected_profit_repeat_current",
            "accepted_repeat_current",
            scale=100.0,
        ),
    ),
    ObservationFeature(
        name="accepted_new_share_current",
        layer="D",
        definition="Share of accepted loans that came from new applicants in the last observed week.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _accepted_share(simulator, "accepted_new_current"),
    ),
    ObservationFeature(
        name="accepted_repeat_share_current",
        layer="D",
        definition="Share of accepted loans that came from repeat applicants in the last observed week.",
        feature_type="segmented",
        normalized=True,
        extractor=lambda simulator: _accepted_share(simulator, "accepted_repeat_current"),
    ),
    ObservationFeature(
        name="reward_roll_mean_4_scaled",
        layer="D",
        definition="Mean weekly reward over the last four interactive weeks scaled by 10,000.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("reward", 0.0)) / 10000.0,
            window=4,
            reducer=np.mean,
        ),
    ),
    ObservationFeature(
        name="reward_roll_std_4_scaled",
        layer="D",
        definition="Standard deviation of weekly reward over the last four interactive weeks scaled by 10,000.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("reward", 0.0)) / 10000.0,
            window=4,
            reducer=np.std,
        ),
    ),
    ObservationFeature(
        name="cumulative_reward_to_date_scaled",
        layer="D",
        definition="Cumulative reward over interactive history scaled by 100,000.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _cumulative_metric(simulator, "reward", scale=100000.0),
    ),
    ObservationFeature(
        name="cumulative_profit_to_date_scaled",
        layer="D",
        definition="Cumulative realized profit over interactive history scaled by 100,000.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _cumulative_metric(simulator, "realized_profit", scale=100000.0),
    ),
    ObservationFeature(
        name="capital_usage_roll_std_4",
        layer="D",
        definition="Standard deviation of projected capital usage over the last four interactive weeks.",
        feature_type="rolling",
        normalized=True,
        extractor=lambda simulator: _rolling_reduce(
            simulator,
            extractor=lambda record: float(record.get("projected_capital_usage_ratio", 0.0)),
            window=4,
            reducer=np.std,
        ),
    ),
    ObservationFeature(
        name="outstanding_ratio_delta_lag_1",
        layer="D",
        definition="Week-over-week change in the outstanding capital ratio.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _record_metric(simulator, 1, "outstanding_ratio")
        - _record_metric(simulator, 2, "outstanding_ratio"),
    ),
    ObservationFeature(
        name="projected_minus_outstanding_gap",
        layer="D",
        definition="Gap between projected capital usage and currently outstanding capital ratio.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _metric(simulator, "projected_capital_usage_ratio") - _metric(simulator, "outstanding_ratio"),
    ),
    ObservationFeature(
        name="threshold_gap_lag_2",
        layer="D",
        definition="Threshold gap observed two interactive weeks ago, normalized by the threshold range.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _threshold_gap_from_record(_interactive_record(simulator, 2), simulator),
    ),
    ObservationFeature(
        name="threshold_gap_delta_lag_1",
        layer="D",
        definition="Week-over-week change in the threshold gap, normalized by the threshold range.",
        feature_type="lagged",
        normalized=True,
        extractor=lambda simulator: _threshold_gap_from_record(_interactive_record(simulator, 1), simulator)
        - _threshold_gap_from_record(_interactive_record(simulator, 2), simulator),
    ),
    ObservationFeature(
        name="applications_ratio_current",
        layer="D",
        definition="Last observed application volume divided by the profile-scaled weekly application baseline.",
        feature_type="scalar",
        normalized=True,
        extractor=lambda simulator: _safe_div(_metric(simulator, "applications_current"), _application_scale(simulator)),
    ),
]


if len(OBSERVATION_FEATURES) != 50:
    raise ValueError(f"Observation registry must contain exactly 50 features, found {len(OBSERVATION_FEATURES)}.")


class ObservationBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.state_dim = int(config["state_dim"])
        if self.state_dim not in ALLOWED_STATE_DIMS:
            raise ValueError(f"Unsupported state_dim={self.state_dim}. Allowed values: {ALLOWED_STATE_DIMS}.")

    @property
    def features(self) -> list[ObservationFeature]:
        return OBSERVATION_FEATURES[: self.state_dim]

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    def build(self, simulator) -> np.ndarray:
        values = []
        for feature in self.features:
            value = float(feature.extractor(simulator))
            if not np.isfinite(value):
                value = 0.0
            values.append(value)
        return np.asarray(values, dtype=np.float32)


def features_for_dimension(state_dim: int) -> list[ObservationFeature]:
    if int(state_dim) not in ALLOWED_STATE_DIMS:
        raise ValueError(f"Unsupported state_dim={state_dim}. Allowed values: {ALLOWED_STATE_DIMS}.")
    return OBSERVATION_FEATURES[: int(state_dim)]


def build_state_dimension_manifest(path: str | Path) -> None:
    path = Path(path)
    lines: list[str] = [
        "# State Dimension Manifest",
        "",
        "Supported state sizes: 12, 20, 30, 50.",
        "",
        "The first 12 features are unchanged baseline features in every state definition.",
        "",
    ]
    previous_dim = 0
    for state_dim in ALLOWED_STATE_DIMS:
        features = features_for_dimension(state_dim)
        new_features = features[previous_dim:state_dim]
        lines.append(f"## {state_dim}D")
        lines.append("")
        lines.append("Baseline compatibility note: the first 12 ordered features are identical to the original 12D baseline.")
        lines.append("")
        lines.append(
            "Newly added at this level: "
            + ", ".join(feature.name for feature in new_features)
            + "."
        )
        lines.append("")
        lines.append("| # | Feature | Layer | Type | Normalized | Definition |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for idx, feature in enumerate(features, start=1):
            normalized = "yes" if feature.normalized else "no"
            lines.append(
                f"| {idx} | `{feature.name}` | {feature.layer} | {feature.feature_type} | {normalized} | {feature.definition} |"
            )
        lines.append("")
        previous_dim = state_dim
    path.write_text("\n".join(lines), encoding="utf-8")
