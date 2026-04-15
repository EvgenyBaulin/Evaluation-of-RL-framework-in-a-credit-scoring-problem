# Dimensionality Experiment Report

## 1. Experiment objective

Measure how observation dimensionality alone changes weekly threshold-controller quality, risk, stability, and behavior in the existing RL credit-scoring simulator.

## 2. What was held constant

- Active profile: `quick`.
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
| 12 | profit_oriented | baseline | 76022.0850 | 88080.5950 |
| 20 | profit_oriented | baseline | 76022.0850 | 88080.5950 |
| 30 | profit_oriented | baseline | 76022.0850 | 88080.5950 |
| 50 | profit_oriented | baseline | 76022.0850 | 88080.5950 |

## 6. Best RL controller for each dimension

| State dim | Best RL | Expected profit | NPV | Cumulative reward | Default rate | Approval rate | Capital usage | Stability index | Threshold volatility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | dqn | 59197.4700 | 57144.6094 | 67308.3592 | 0.1233 | 0.4159 | 0.4454 | 0.0001 | 0.0000 |
| 20 | double_dqn | 61096.4396 | 58978.9759 | 69812.0186 | 0.1260 | 0.4329 | 0.4569 | 0.0001 | 0.0000 |
| 30 | dqn | 66758.2865 | 64755.0731 | 78219.6900 | 0.1185 | 0.3527 | 0.3902 | 0.0001 | 0.0000 |
| 50 | dqn | 66946.8121 | 64815.0538 | 78198.3921 | 0.1195 | 0.3935 | 0.4280 | 0.0001 | 0.0000 |

## 7. Cross-dimension comparison

- expected profit: 12->20: 1898.97, 20->30: 5661.85, 30->50: 188.53
- NPV: 12->20: 1834.37, 20->30: 5776.10, 30->50: 59.98
- cumulative reward: 12->20: 2503.66, 20->30: 8407.67, 30->50: -21.30
- default rate: 12->20: 0.00, 20->30: -0.01, 30->50: 0.00
- approval rate: 12->20: 0.02, 20->30: -0.08, 30->50: 0.04
- capital usage: 12->20: 0.01, 20->30: -0.07, 30->50: 0.04
- stability index: 12->20: -0.00, 20->30: 0.00, 30->50: -0.00
- threshold volatility: 12->20: 0.00, 20->30: 0.00, 30->50: 0.00

## 8. Saturation assessment

- Gains flatten by 30 to 50 dimensions.

## 9. Signal vs complexity assessment

- Larger states add usable signal in this run: the best dimension improves profit without increasing default rate or threshold volatility.

## 10. Validity checks

| Check | Status | Detail |
| --- | --- | --- |
| First 12 features unchanged | PASS | Programmatic check passed on reset plus three fixed-action transitions. |
| Controlled protocol consistency | PASS | Seeds, scenarios, reward settings, controller sets, and evaluation protocol match across dimensions. |
| No future leakage in added features | PASS | ObservationBuilder reads only interactive history, last_week_metrics, interactive_week, and fixed config constants. |
| Controller coverage for dim 12 | PASS | Every controller completed 72 evaluation runs. |
| Output files for dim 12 | PASS | Required CSVs and plots exist and are non-empty. |
| Controller coverage for dim 20 | PASS | Every controller completed 72 evaluation runs. |
| Output files for dim 20 | PASS | Required CSVs and plots exist and are non-empty. |
| Controller coverage for dim 30 | PASS | Every controller completed 72 evaluation runs. |
| Output files for dim 30 | PASS | Required CSVs and plots exist and are non-empty. |
| Controller coverage for dim 50 | PASS | Every controller completed 72 evaluation runs. |
| Output files for dim 50 | PASS | Required CSVs and plots exist and are non-empty. |
| Cross-dimension internal consistency | PASS | Best-row extracts match the global comparison summary. |
| Run failures | PASS | No dimension run failed. |

## 11. Final conclusion

- Successful dimensions: [12, 20, 30, 50].
- Gains flatten by 30 to 50 dimensions.
- Larger states add usable signal in this run: the best dimension improves profit without increasing default rate or threshold volatility.
- Use the per-dimension folders under `outputs/` for the full CSV and plot set, and the cross-dimension files in `outputs/` for thesis-style comparison figures.