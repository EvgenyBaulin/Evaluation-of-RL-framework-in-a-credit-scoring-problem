# Dimensionality Experiment Report

## 1. Experiment objective

Measure how observation dimensionality alone changes weekly threshold-controller quality, risk, stability, and behavior in the existing RL credit-scoring simulator.

## 2. What was held constant

- Active profile: `full`.
- Environment dynamics, weekly interaction logic, action semantics, threshold ranges, reward definition, delayed reward mechanism, delayed outcome mechanism, warm-up logic, terminal settlement logic, train/eval protocol, seeds, scenarios, bootstrap CI settings, controller set, and baseline set were held fixed across every run.
- Selection rule for `best overall controller` and `best RL controller`: highest `cumulative_reward_mean`, then highest `expected_profit_mean`, then highest `stability_index_mean`, then lowest `default_rate_mean`.

## 3. What changed

- Only `state_dim` changed, with the controlled values `12`, `20`, `30`, and `50`.
- Model input layers changed only through the existing `obs_dim` wiring already used by DQN / Double-DQN and SB3 policies.

## 4. Exact feature composition

The first 12 ordered features are unchanged baseline features in all four configurations.

- 12D: `week_progress`, `approval_rate_current`, `approval_rate_new`, `approval_rate_repeat`, `rolling_realized_default_rate`, `expected_default_rate_current`, `realized_profit_scaled`, `rolling_profit_volatility_scaled`, `projected_capital_usage_ratio`, `outstanding_ratio`, `threshold_new_normalized`, `threshold_repeat_normalized`
- 20D: `week_progress`, `approval_rate_current`, `approval_rate_new`, `approval_rate_repeat`, `rolling_realized_default_rate`, `expected_default_rate_current`, `realized_profit_scaled`, `rolling_profit_volatility_scaled`, `projected_capital_usage_ratio`, `outstanding_ratio`, `threshold_new_normalized`, `threshold_repeat_normalized`, `repeat_share_current`, `expected_profit_per_application_scaled`, `expected_npv_per_application_scaled`, `realized_npv_scaled`, `weekly_reward_scaled`, `threshold_gap_normalized`, `capital_headroom_ratio`, `realized_expected_default_gap`
- 30D: `week_progress`, `approval_rate_current`, `approval_rate_new`, `approval_rate_repeat`, `rolling_realized_default_rate`, `expected_default_rate_current`, `realized_profit_scaled`, `rolling_profit_volatility_scaled`, `projected_capital_usage_ratio`, `outstanding_ratio`, `threshold_new_normalized`, `threshold_repeat_normalized`, `repeat_share_current`, `expected_profit_per_application_scaled`, `expected_npv_per_application_scaled`, `realized_npv_scaled`, `weekly_reward_scaled`, `threshold_gap_normalized`, `capital_headroom_ratio`, `realized_expected_default_gap`, `approval_rate_lag_2`, `realized_default_rate_lag_2`, `realized_profit_lag_2_scaled`, `capital_usage_lag_2`, `approval_rate_roll_mean_4`, `realized_default_rate_roll_mean_4`, `realized_profit_roll_mean_4_scaled`, `realized_profit_roll_std_4_scaled`, `threshold_new_delta_lag_1`, `threshold_repeat_delta_lag_1`
- 50D: `week_progress`, `approval_rate_current`, `approval_rate_new`, `approval_rate_repeat`, `rolling_realized_default_rate`, `expected_default_rate_current`, `realized_profit_scaled`, `rolling_profit_volatility_scaled`, `projected_capital_usage_ratio`, `outstanding_ratio`, `threshold_new_normalized`, `threshold_repeat_normalized`, `repeat_share_current`, `expected_profit_per_application_scaled`, `expected_npv_per_application_scaled`, `realized_npv_scaled`, `weekly_reward_scaled`, `threshold_gap_normalized`, `capital_headroom_ratio`, `realized_expected_default_gap`, `approval_rate_lag_2`, `realized_default_rate_lag_2`, `realized_profit_lag_2_scaled`, `capital_usage_lag_2`, `approval_rate_roll_mean_4`, `realized_default_rate_roll_mean_4`, `realized_profit_roll_mean_4_scaled`, `realized_profit_roll_std_4_scaled`, `threshold_new_delta_lag_1`, `threshold_repeat_delta_lag_1`, `approval_rate_new_roll_mean_4`, `approval_rate_repeat_roll_mean_4`, `approval_rate_new_lag_2`, `approval_rate_repeat_lag_2`, `expected_default_rate_new_current`, `expected_default_rate_repeat_current`, `expected_profit_new_per_accept_scaled`, `expected_profit_repeat_per_accept_scaled`, `accepted_new_share_current`, `accepted_repeat_share_current`, `reward_roll_mean_4_scaled`, `reward_roll_std_4_scaled`, `cumulative_reward_to_date_scaled`, `cumulative_profit_to_date_scaled`, `capital_usage_roll_std_4`, `outstanding_ratio_delta_lag_1`, `projected_minus_outstanding_gap`, `threshold_gap_lag_2`, `threshold_gap_delta_lag_1`, `applications_ratio_current`

Detailed one-line definitions, types, normalization flags, and incremental additions are recorded in `state_dimension_manifest.md`.

## 5. Best overall controller for each dimension

| State dim | Controller | Type | Expected profit | Cumulative reward |
| --- | --- | --- | --- | --- |
| 12 | constraint_aware_weekly | baseline | 3802547.1621 | 3666052.3089 |
| 20 | constraint_aware_weekly | baseline | 3802547.1621 | 3666052.3089 |
| 30 | constraint_aware_weekly | baseline | 3802547.1621 | 3666052.3089 |
| 50 | constraint_aware_weekly | baseline | 3802547.1621 | 3666052.3089 |

## 6. Best RL controller for each dimension

| State dim | Best RL | Expected profit | NPV | Cumulative reward | Default rate | Approval rate | Capital usage | Stability index | Threshold volatility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | sac | 3785010.1389 | 3699360.4215 | 3631406.0246 | 0.0768 | 0.9515 | 3.1269 | 0.0000 | 2.1038 |
| 20 | a2c | 3761561.7091 | 3676533.6738 | 3624570.0728 | 0.0768 | 0.9439 | 3.0995 | 0.0000 | 1.2707 |
| 30 | dqn | 3706594.0438 | 3623193.3248 | 3633944.5093 | 0.0749 | 0.9137 | 3.0106 | 0.0000 | 0.0000 |
| 50 | sac | 3724498.5969 | 3640698.2243 | 3605339.3548 | 0.0762 | 0.9320 | 3.0630 | 0.0000 | 4.2004 |

## 7. Cross-dimension comparison

- expected profit: 12->20: -23448.43, 20->30: -54967.67, 30->50: 17904.55
- NPV: 12->20: -22826.75, 20->30: -53340.35, 30->50: 17504.90
- cumulative reward: 12->20: -6835.95, 20->30: 9374.44, 30->50: -28605.15
- default rate: 12->20: -0.00, 20->30: -0.00, 30->50: 0.00
- approval rate: 12->20: -0.01, 20->30: -0.03, 30->50: 0.02
- capital usage: 12->20: -0.03, 20->30: -0.09, 30->50: 0.05
- stability index: 12->20: -0.00, 20->30: 0.00, 30->50: -0.00
- threshold volatility: 12->20: -0.83, 20->30: -1.27, 30->50: 4.20

## 8. Saturation assessment

- Expected profit peaks at 12 dimensions and then weakens at higher dimensionality.

## 9. Signal vs complexity assessment

- Larger states mostly add complexity in this run: added dimensions do not pay for themselves on profit-risk behavior.

## 10. Validity checks

| Check | Status | Detail |
| --- | --- | --- |
| First 12 features unchanged | PASS | Programmatic check passed on reset plus three fixed-action transitions. |
| Controlled protocol consistency | PASS | Seeds, scenarios, reward settings, controller sets, and evaluation protocol match across dimensions. |
| No future leakage in added features | PASS | ObservationBuilder reads only interactive history, last_week_metrics, interactive_week, and fixed config constants. |
| Controller coverage for dim 12 | PASS | Every controller completed 576 evaluation runs. |
| Output files for dim 12 | PASS | Required CSVs and plots exist and are non-empty. |
| Controller coverage for dim 20 | PASS | Every controller completed 576 evaluation runs. |
| Output files for dim 20 | PASS | Required CSVs and plots exist and are non-empty. |
| Controller coverage for dim 30 | PASS | Every controller completed 576 evaluation runs. |
| Output files for dim 30 | PASS | Required CSVs and plots exist and are non-empty. |
| Controller coverage for dim 50 | PASS | Every controller completed 576 evaluation runs. |
| Output files for dim 50 | PASS | Required CSVs and plots exist and are non-empty. |
| Cross-dimension internal consistency | PASS | Best-row extracts match the global comparison summary. |
| Run failures | PASS | No dimension run failed. |

## 11. Final conclusion

- Successful dimensions: [12, 20, 30, 50].
- Expected profit peaks at 12 dimensions and then weakens at higher dimensionality.
- Larger states mostly add complexity in this run: added dimensions do not pay for themselves on profit-risk behavior.
- Use the per-dimension folders under `outputs/` for the full CSV and plot set, and the cross-dimension files in `outputs/` for thesis-style comparison figures.