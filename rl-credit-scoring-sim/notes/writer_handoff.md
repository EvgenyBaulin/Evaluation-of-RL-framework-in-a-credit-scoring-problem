# Writer Handoff

## 1. Repository focus

The repository is now centered on a controlled state-dimensionality experiment for the weekly threshold-control RL framework.
- Active comparison profile: `quick`.
- Only `state_dim` changes across runs: `12`, `20`, `30`, `50`.
- Environment dynamics, weekly interaction logic, action semantics, threshold ranges, reward definition, delayed reward logic, delayed outcomes, warm-up, terminal settlement, seeds, scenarios, bootstrap settings, metrics, controller set, and baseline set are held fixed.
- The first 12 ordered features are unchanged baseline features in every state definition.

## 2. How to run it

- Main one-command experiment: `python scripts/run_dimensionality_experiment.py --profile quick`.
- CLI equivalent: `python scripts/run_pipeline.py --dimensionality-experiment --profile quick`.
- Single-dimension smoke test: `python scripts/run_pipeline.py --state-dim 20 --profile quick`.

## 3. Feature layers

- 12D baseline: `week_progress`, `approval_rate_current`, `approval_rate_new`, `approval_rate_repeat`, `rolling_realized_default_rate`, `expected_default_rate_current`, `realized_profit_scaled`, `rolling_profit_volatility_scaled`, `projected_capital_usage_ratio`, `outstanding_ratio`, `threshold_new_normalized`, `threshold_repeat_normalized`
- 20D adds B-layer features on top of the unchanged baseline: `repeat_share_current`, `expected_profit_per_application_scaled`, `expected_npv_per_application_scaled`, `realized_npv_scaled`, `weekly_reward_scaled`, `threshold_gap_normalized`, `capital_headroom_ratio`, `realized_expected_default_gap`
- 30D adds C-layer features on top of 20D: `approval_rate_lag_2`, `realized_default_rate_lag_2`, `realized_profit_lag_2_scaled`, `capital_usage_lag_2`, `approval_rate_roll_mean_4`, `realized_default_rate_roll_mean_4`, `realized_profit_roll_mean_4_scaled`, `realized_profit_roll_std_4_scaled`, `threshold_new_delta_lag_1`, `threshold_repeat_delta_lag_1`
- 50D adds D-layer features on top of 30D: `approval_rate_new_roll_mean_4`, `approval_rate_repeat_roll_mean_4`, `approval_rate_new_lag_2`, `approval_rate_repeat_lag_2`, `expected_default_rate_new_current`, `expected_default_rate_repeat_current`, `expected_profit_new_per_accept_scaled`, `expected_profit_repeat_per_accept_scaled`, `accepted_new_share_current`, `accepted_repeat_share_current`, `reward_roll_mean_4_scaled`, `reward_roll_std_4_scaled`, `cumulative_reward_to_date_scaled`, `cumulative_profit_to_date_scaled`, `capital_usage_roll_std_4`, `outstanding_ratio_delta_lag_1`, `projected_minus_outstanding_gap`, `threshold_gap_lag_2`, `threshold_gap_delta_lag_1`, `applications_ratio_current`

Detailed one-line definitions, feature types, normalization flags, and ordering are in `state_dimension_manifest.md`.

## 4. Best controllers by dimension

Best overall controller per dimension:

| State dim | Best overall | Type | Expected profit | Cumulative reward |
| --- | --- | --- | --- | --- |
| 12 | profit_oriented | baseline | 76022.0850 | 88080.5950 |
| 20 | profit_oriented | baseline | 76022.0850 | 88080.5950 |
| 30 | profit_oriented | baseline | 76022.0850 | 88080.5950 |
| 50 | profit_oriented | baseline | 76022.0850 | 88080.5950 |

Best RL controller per dimension:

| State dim | Best RL | Expected profit | NPV | Cumulative reward | Default rate | Approval rate | Capital usage | Stability index | Threshold volatility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | dqn | 59197.4700 | 57144.6094 | 67308.3592 | 0.1233 | 0.4159 | 0.4454 | 0.0001 | 0.0000 |
| 20 | double_dqn | 61096.4396 | 58978.9759 | 69812.0186 | 0.1260 | 0.4329 | 0.4569 | 0.0001 | 0.0000 |
| 30 | dqn | 66758.2865 | 64755.0731 | 78219.6900 | 0.1185 | 0.3527 | 0.3902 | 0.0001 | 0.0000 |
| 50 | dqn | 66946.8121 | 64815.0538 | 78198.3921 | 0.1195 | 0.3935 | 0.4280 | 0.0001 | 0.0000 |

## 5. Main interpretation

- Gains flatten by 30 to 50 dimensions.
- Larger states add usable signal in this run: the best dimension improves profit without increasing default rate or threshold volatility.
- In the current quick-profile outputs, the strongest overall controller is the same rule-based baseline across all four dimensions, while the best RL controller improves materially from 12D to 30D and then largely plateaus at 50D.

## 6. Figure map

- `outputs/dim_12/expected_profit_by_scenario_dim12.png`, `outputs/dim_20/expected_profit_by_scenario_dim20.png`, `outputs/dim_30/expected_profit_by_scenario_dim30.png`, `outputs/dim_50/expected_profit_by_scenario_dim50.png`: per-dimension expected-profit comparison by scenario.
- `outputs/dim_12/cumulative_reward_curves_dim12.png`, `outputs/dim_20/cumulative_reward_curves_dim20.png`, `outputs/dim_30/cumulative_reward_curves_dim30.png`, `outputs/dim_50/cumulative_reward_curves_dim50.png`: weekly cumulative reward trajectories for each dimension.
- `outputs/dim_12/cumulative_profit_curves_dim12.png`, `outputs/dim_20/cumulative_profit_curves_dim20.png`, `outputs/dim_30/cumulative_profit_curves_dim30.png`, `outputs/dim_50/cumulative_profit_curves_dim50.png`: weekly cumulative profit trajectories for each dimension.
- `outputs/dim_12/default_rate_by_scenario_dim12.png`, `outputs/dim_20/default_rate_by_scenario_dim20.png`, `outputs/dim_30/default_rate_by_scenario_dim30.png`, `outputs/dim_50/default_rate_by_scenario_dim50.png`: per-dimension default-rate comparison by scenario.
- `outputs/dim_12/approval_rate_by_scenario_dim12.png`, `outputs/dim_20/approval_rate_by_scenario_dim20.png`, `outputs/dim_30/approval_rate_by_scenario_dim30.png`, `outputs/dim_50/approval_rate_by_scenario_dim50.png`: per-dimension approval-rate comparison by scenario.
- `outputs/dim_12/capital_usage_by_scenario_dim12.png`, `outputs/dim_20/capital_usage_by_scenario_dim20.png`, `outputs/dim_30/capital_usage_by_scenario_dim30.png`, `outputs/dim_50/capital_usage_by_scenario_dim50.png`: per-dimension capital-usage comparison by scenario.
- `outputs/dim_12/threshold_volatility_dim12.png`, `outputs/dim_20/threshold_volatility_dim20.png`, `outputs/dim_30/threshold_volatility_dim30.png`, `outputs/dim_50/threshold_volatility_dim50.png`: per-dimension threshold-volatility comparison by scenario.
- `outputs/dim_12/best_rl_threshold_paths_dim12.png`, `outputs/dim_20/best_rl_threshold_paths_dim20.png`, `outputs/dim_30/best_rl_threshold_paths_dim30.png`, `outputs/dim_50/best_rl_threshold_paths_dim50.png`: threshold paths of the strongest RL controller within each dimension.
- `outputs/metric_vs_dimension_expected_profit.png`, `outputs/metric_vs_dimension_npv.png`, `outputs/metric_vs_dimension_cumulative_reward.png`, `outputs/metric_vs_dimension_default_rate.png`, `outputs/metric_vs_dimension_stability_index.png`: cross-dimension comparison figures for thesis-style interpretation.

## 7. Table map

- `outputs/dim_12/main_scenario_summary.csv`, `outputs/dim_20/main_scenario_summary.csv`, `outputs/dim_30/main_scenario_summary.csv`, `outputs/dim_50/main_scenario_summary.csv`: per-dimension bootstrap CI summary by controller and scenario.
- `outputs/dim_12/main_overall_summary.csv`, `outputs/dim_20/main_overall_summary.csv`, `outputs/dim_30/main_overall_summary.csv`, `outputs/dim_50/main_overall_summary.csv`: per-dimension overall ranking across all scenarios.
- `outputs/dim_12/run_level_metrics.csv`, `outputs/dim_20/run_level_metrics.csv`, `outputs/dim_30/run_level_metrics.csv`, `outputs/dim_50/run_level_metrics.csv`: per-run metrics before aggregation.
- `outputs/dim_12/weekly_run_metrics.csv`, `outputs/dim_20/weekly_run_metrics.csv`, `outputs/dim_30/weekly_run_metrics.csv`, `outputs/dim_50/weekly_run_metrics.csv`: per-week metrics before aggregation.
- `outputs/dim_12/weekly_reward_curves.csv`, `outputs/dim_20/weekly_reward_curves.csv`, `outputs/dim_30/weekly_reward_curves.csv`, `outputs/dim_50/weekly_reward_curves.csv`: cumulative reward curves with bootstrap bands.
- `outputs/dim_12/weekly_profit_curves.csv`, `outputs/dim_20/weekly_profit_curves.csv`, `outputs/dim_30/weekly_profit_curves.csv`, `outputs/dim_50/weekly_profit_curves.csv`: cumulative profit curves with bootstrap bands.
- `outputs/dimension_comparison_summary.csv`: all controllers stacked across dimensions for cross-dimension analysis.
- `outputs/best_rl_by_dimension.csv`: best RL controller row for each state size.
- `outputs/overall_best_by_dimension.csv`: best overall controller row for each state size.

## 8. Validity checks

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

## 9. Writing guidance

- Use `dimensionality_experiment_report.md` for the compact conclusion and metric deltas.
- Use `state_dimension_manifest.md` when describing the exact observation composition.
- Use the per-dimension `outputs/dim_*` folders for figure-by-figure discussion and the root `outputs/metric_vs_dimension_*.png` figures for the headline thesis comparison.
- Report explicitly that the first 12 baseline features were checked programmatically to remain unchanged across all four configurations.