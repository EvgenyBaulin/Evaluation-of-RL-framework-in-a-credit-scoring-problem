# 1. Project overview

- Project topic: evaluation of an RL framework for weekly acceptance-threshold control in credit scoring.
- Goal: compare dynamic RL threshold policies against static and rule-based baselines in a simulation-only setting with delayed loan outcomes and configurable constraints.
- This handoff corresponds to the `30D` pipeline output written under `outputs/dim_30`.
- Problem setup: the controller selects weekly thresholds, not per-application approve/reject decisions; two thresholds are supported simultaneously for new and repeat clients.
- What was taken from the reference project: weekly interaction, threshold control, delayed rewards, delayed outcomes, warm-up logic, and scenario-based shift analysis.
- What was extended: split-policy control as the default design, configurable constraint-aware reward shaping, multi-agent comparison across DQN/Double-DQN/PPO/SAC, multi-seed confidence intervals, structured ablations, and a writer handoff.

# 2. Environment design

- State definition: 30-dimensional observation. The first 12 ordered features remain the baseline features: `week_progress`, `approval_rate_current`, `approval_rate_new`, `approval_rate_repeat`, `rolling_realized_default_rate`, `expected_default_rate_current`, `realized_profit_scaled`, `rolling_profit_volatility_scaled`, `projected_capital_usage_ratio`, `outstanding_ratio`, `threshold_new_normalized`, `threshold_repeat_normalized`.
- Added beyond the baseline for this run: `repeat_share_current`, `expected_profit_per_application_scaled`, `expected_npv_per_application_scaled`, `realized_npv_scaled`, `weekly_reward_scaled`, `threshold_gap_normalized`, `capital_headroom_ratio`, `realized_expected_default_gap`, `approval_rate_lag_2`, `realized_default_rate_lag_2`, `realized_profit_lag_2_scaled`, `capital_usage_lag_2`, `approval_rate_roll_mean_4`, `realized_default_rate_roll_mean_4`, `realized_profit_roll_mean_4_scaled`, `realized_profit_roll_std_4_scaled`, `threshold_new_delta_lag_1`, `threshold_repeat_delta_lag_1`. The cumulative layer definitions are recorded in `state_dimension_manifest.md`.
- Action definition: weekly thresholds for new and repeat clients. Discrete agents act on a threshold grid from 35 to 85 with step 5; PPO and SAC use continuous threshold control mapped to the same range.
- Weekly threshold logic: applications are generated each week, accepted if score >= threshold, and then scheduled into future cash-flow events.
- Delayed reward mechanism: with delayed reward enabled, reward is driven by realized portfolio cash flows from loans originated earlier plus configurable shaping terms and penalties.
- Delayed outcome mechanism: each accepted loan schedules a repayment, default-loss, and optionally a later recovery event; a terminal settlement closes remaining pending events at episode end to avoid truncation bias.
- Split between new and repeat policies: the simulator keeps separate segment score/default processes and allows threshold pairs; an ablation disables this and forces a shared threshold.
- Configurable constraints: default rate, approval rate, capital usage, and volatility are configurable in `configs/run_profile.yaml`.

# 3. Baseline methods

- `static_threshold`: fixed shared threshold benchmark.
- `split_policy_static`: fixed but segment-specific thresholds using the default new/repeat policy.
- `profit_oriented`: weekly grid search that maximizes expected current-cohort profit and intentionally ignores broader constraints.
- `risk_aware_weekly`: hand-crafted feedback rule that raises or lowers thresholds using rolling realized default rate, approval pressure, and capital usage.
- `constraint_aware_weekly`: weekly grid search over threshold pairs that maximizes expected profit subject to penalty-weighted constraint violations.
- Why they are needed: together they cover static, myopic, reactive, and explicitly constrained non-RL alternatives.

# 4. RL methods

- Implemented agents: DQN, Double-DQN, PPO, and SAC.
- Why these agents were chosen: DQN/Double-DQN cover value-based discrete threshold control; PPO and SAC cover continuous policy optimization and actor-critic learning.
- Action spaces: DQN/Double-DQN use the discrete threshold grid; PPO/SAC use continuous threshold pairs mapped back onto the same threshold range.
- Training settings: profile-driven. Quick profile used 12 training episodes, 26 interactive weeks, seeds [11, 23, 37], and scenario randomization set to True.

# 5. Reward formulation

- Base reward: `profit_weight * profit + npv_weight * npv`.
- Risk-aware shaping: optional dense penalty on expected current default rate.
- Constraint penalties:
  - default rate penalty against target 0.12
  - approval rate penalty against target 0.42
  - capital usage penalty against target 0.82
  - volatility penalty against target 8500.0
- Trade-off logic: the reward balances portfolio profitability against stability and resource constraints instead of maximizing nominal short-term gain only.

# 6. Experimental scenarios

- `base_market`: Stable market with moderate approval and default dynamics.
- `drift`: Gradual deterioration in new-client scores and mild volume growth.
- `adverse_stress`: Abrupt negative shock with higher defaults and capital pressure.
- `class_imbalance_shift`: Faster growth in new-client share with lower average quality.
- `noise`: Higher measurement noise with unstable weekly application volumes.
- `split_policy_dynamics`: New clients deteriorate while repeat clients improve, making split thresholds valuable.

# 7. Metrics

- Approval rate: accepted applications / total applications over the episode.
- Default rate: realized default-loss events / accepted applications.
- Expected profit: total realized profit across interactive weeks plus terminal settlement.
- NPV: discounted realized portfolio profit.
- Cumulative reward: total shaped RL reward.
- Capital usage mean: average projected capital usage ratio.
- Reward volatility: standard deviation of weekly rewards.
- Stability index: `1 / (1 + reward_volatility)` in this implementation.
- Threshold volatility: standard deviation of weekly threshold changes.
- Primary metrics: expected profit, NPV, cumulative reward, and default rate.

# 8. Experimental protocol

- Split logic: simulation-only; warm-up weeks initialize the portfolio, interactive weeks produce policy decisions, and terminal settlement closes remaining pending loans.
- Controlled dimensionality note: when the repository-level comparison is launched, the same code is rerun for `state_dim` = 12, 20, 30, and 50 while holding all other settings fixed.
- Seeds: [11, 23, 37].
- Confidence intervals: bootstrap with confidence level 0.95 and 120 resamples under the active profile.
- Hyperparameter strategy: fixed profile-driven defaults rather than per-scenario retuning.
- Evaluation procedure: every controller is tested on every scenario for 4 runs per seed; CI tables and CI time-series curves are exported.

# 9. Results

- Best overall controller: `profit_oriented` with cumulative reward mean 88080.60.
- Best RL controller: `dqn` with cumulative reward mean 78219.69 and expected profit mean 66758.29.
- Scenario winners:
- `adverse_stress`: `profit_oriented` with expected profit mean 29336.42
- `base_market`: `double_dqn` with expected profit mean 86747.36
- `class_imbalance_shift`: `constraint_aware_weekly` with expected profit mean 70649.29
- `drift`: `profit_oriented` with expected profit mean 59526.10
- `noise`: `risk_aware_weekly` with expected profit mean 97377.95
- `split_policy_dynamics`: `profit_oriented` with expected profit mean 120324.99
- Stability: see `expected_profit_seed_std` and `cumulative_reward_seed_std` columns in `main_scenario_summary.csv` and `main_overall_summary.csv`.

# 10. Ablations

Not run.


# 11. Thesis novelty

- Core thesis idea: a controller can sacrifice local short-run profit to improve end-of-horizon portfolio quality.
- Diagnostic results:
- `dqn` in `adverse_stress`: early gap -643.46, final gap 250578.20
- `double_dqn` in `adverse_stress`: early gap -449.55, final gap 214106.86
- `ppo` in `adverse_stress`: early gap -2010.07, final gap 114364.76
- Use `locally_worse_globally_better.csv` and, when available, `locally_worse_globally_better.png` to support this claim.

# 12. Limitations

- The simulator is synthetic and parameterized, not calibrated on confidential production logs.
- Reward shaping and scenario design are controlled experiments, not causal identification.
- RL training horizons in the quick profile are intentionally small and should not be over-claimed as final convergence evidence.
- Confidence intervals quantify simulation uncertainty under the chosen generator, not external validity.

# 13. Figure map

- `expected_profit_by_scenario.png`: Expected profit comparison across controllers and scenarios. | thesis sections: experiments/results
- `cumulative_reward_curves.png`: Weekly cumulative reward paths with confidence intervals. | thesis sections: experiments/results
- `cumulative_profit_curves.png`: Weekly cumulative portfolio profit paths with confidence intervals. | thesis sections: experiments/results
- `split_policy_threshold_paths.png`: New and repeat threshold trajectories in the split-policy scenario. | thesis sections: experiments/results
- `locally_worse_globally_better.png`: Example of an early-loss / late-gain portfolio trajectory. | thesis sections: experiments/results

# 14. Table map

- `main_scenario_summary.csv`: Bootstrap CI summary for every controller-scenario pair. | thesis sections: methods/results/appendix
- `main_overall_summary.csv`: Overall controller ranking across all scenarios. | thesis sections: methods/results/appendix
- `weekly_reward_curves.csv`: Weekly cumulative reward curves with CI envelopes. | thesis sections: methods/results/appendix
- `weekly_profit_curves.csv`: Weekly cumulative profit curves with CI envelopes. | thesis sections: methods/results/appendix
- `run_level_metrics.csv`: Per-run metrics before aggregation. | thesis sections: methods/results/appendix
- `weekly_run_metrics.csv`: Per-week metrics before aggregation. | thesis sections: methods/results/appendix
- `locally_worse_globally_better.csv`: Early-loss / late-gain diagnostic table. | thesis sections: methods/results/appendix

# 15. Ready-to-use writing fragments

- "The controller does not approve or reject individual applications directly; instead, it updates weekly score cutoffs for new and repeat clients."
- "The simulation includes delayed credit outcomes, so portfolio profit is realized after repayment, default, and late-recovery events rather than immediately at origination."
- "Constraint-aware reward shaping was used to balance profitability against default risk, approval pressure, capital usage, and reward volatility."
- "Bootstrap confidence intervals were computed for both aggregate outcome tables and weekly trajectory plots, ensuring uncertainty quantification throughout the evaluation pipeline."
- "The split-policy scenario demonstrates why a single shared threshold can be suboptimal when new-client and repeat-client dynamics diverge."
