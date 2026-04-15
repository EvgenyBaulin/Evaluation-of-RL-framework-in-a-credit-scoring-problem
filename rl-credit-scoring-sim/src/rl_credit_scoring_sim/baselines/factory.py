from __future__ import annotations

from rl_credit_scoring_sim.baselines.policies import (
    ConstraintAwareWeeklyBaseline,
    ProfitOrientedBaseline,
    RiskAwareWeeklyBaseline,
    SplitPolicyStaticBaseline,
    StaticThresholdBaseline,
)


def make_baseline(name: str, config: dict):
    registry = {
        "static_threshold": StaticThresholdBaseline,
        "profit_oriented": ProfitOrientedBaseline,
        "risk_aware_weekly": RiskAwareWeeklyBaseline,
        "constraint_aware_weekly": ConstraintAwareWeeklyBaseline,
        "split_policy_static": SplitPolicyStaticBaseline,
    }
    if name not in registry:
        raise KeyError(f"Unknown baseline: {name}")
    return registry[name](config)
