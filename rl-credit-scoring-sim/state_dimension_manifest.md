# State Dimension Manifest

Supported state sizes: 12, 20, 30, 50.

The first 12 features are unchanged baseline features in every state definition.

## 12D

Baseline compatibility note: the first 12 ordered features are identical to the original 12D baseline.

Newly added at this level: week_progress, approval_rate_current, approval_rate_new, approval_rate_repeat, rolling_realized_default_rate, expected_default_rate_current, realized_profit_scaled, rolling_profit_volatility_scaled, projected_capital_usage_ratio, outstanding_ratio, threshold_new_normalized, threshold_repeat_normalized.

| # | Feature | Layer | Type | Normalized | Definition |
| --- | --- | --- | --- | --- | --- |
| 1 | `week_progress` | A | scalar | yes | Interactive week index divided by the episode horizon. |
| 2 | `approval_rate_current` | A | scalar | yes | Last observed overall approval rate. |
| 3 | `approval_rate_new` | A | segmented | yes | Last observed approval rate for new applicants. |
| 4 | `approval_rate_repeat` | A | segmented | yes | Last observed approval rate for repeat applicants. |
| 5 | `rolling_realized_default_rate` | A | rolling | yes | Rolling realized default rate from the simulator reward window. |
| 6 | `expected_default_rate_current` | A | scalar | yes | Last observed expected default rate among accepted applications. |
| 7 | `realized_profit_scaled` | A | scalar | yes | Last observed realized profit scaled by 10,000. |
| 8 | `rolling_profit_volatility_scaled` | A | rolling | yes | Rolling realized profit volatility scaled by 10,000. |
| 9 | `projected_capital_usage_ratio` | A | scalar | yes | Projected capital usage ratio after the last action. |
| 10 | `outstanding_ratio` | A | scalar | yes | Outstanding principal divided by the capital limit. |
| 11 | `threshold_new_normalized` | A | scalar | yes | Last applied new-client threshold normalized to the configured threshold range. |
| 12 | `threshold_repeat_normalized` | A | scalar | yes | Last applied repeat-client threshold normalized to the configured threshold range. |

## 20D

Baseline compatibility note: the first 12 ordered features are identical to the original 12D baseline.

Newly added at this level: repeat_share_current, expected_profit_per_application_scaled, expected_npv_per_application_scaled, realized_npv_scaled, weekly_reward_scaled, threshold_gap_normalized, capital_headroom_ratio, realized_expected_default_gap.

| # | Feature | Layer | Type | Normalized | Definition |
| --- | --- | --- | --- | --- | --- |
| 1 | `week_progress` | A | scalar | yes | Interactive week index divided by the episode horizon. |
| 2 | `approval_rate_current` | A | scalar | yes | Last observed overall approval rate. |
| 3 | `approval_rate_new` | A | segmented | yes | Last observed approval rate for new applicants. |
| 4 | `approval_rate_repeat` | A | segmented | yes | Last observed approval rate for repeat applicants. |
| 5 | `rolling_realized_default_rate` | A | rolling | yes | Rolling realized default rate from the simulator reward window. |
| 6 | `expected_default_rate_current` | A | scalar | yes | Last observed expected default rate among accepted applications. |
| 7 | `realized_profit_scaled` | A | scalar | yes | Last observed realized profit scaled by 10,000. |
| 8 | `rolling_profit_volatility_scaled` | A | rolling | yes | Rolling realized profit volatility scaled by 10,000. |
| 9 | `projected_capital_usage_ratio` | A | scalar | yes | Projected capital usage ratio after the last action. |
| 10 | `outstanding_ratio` | A | scalar | yes | Outstanding principal divided by the capital limit. |
| 11 | `threshold_new_normalized` | A | scalar | yes | Last applied new-client threshold normalized to the configured threshold range. |
| 12 | `threshold_repeat_normalized` | A | scalar | yes | Last applied repeat-client threshold normalized to the configured threshold range. |
| 13 | `repeat_share_current` | B | segmented | yes | Share of applications coming from repeat clients in the last observed week. |
| 14 | `expected_profit_per_application_scaled` | B | scalar | yes | Last observed expected cohort profit per application scaled by 100. |
| 15 | `expected_npv_per_application_scaled` | B | scalar | yes | Last observed expected cohort NPV per application scaled by 100. |
| 16 | `realized_npv_scaled` | B | scalar | yes | Last observed realized NPV scaled by 10,000. |
| 17 | `weekly_reward_scaled` | B | scalar | yes | Last observed weekly reward scaled by 10,000. |
| 18 | `threshold_gap_normalized` | B | scalar | yes | Difference between new and repeat thresholds normalized by the threshold range. |
| 19 | `capital_headroom_ratio` | B | scalar | yes | Remaining capital headroom after the last action. |
| 20 | `realized_expected_default_gap` | B | scalar | yes | Gap between rolling realized default rate and last observed expected default rate. |

## 30D

Baseline compatibility note: the first 12 ordered features are identical to the original 12D baseline.

Newly added at this level: approval_rate_lag_2, realized_default_rate_lag_2, realized_profit_lag_2_scaled, capital_usage_lag_2, approval_rate_roll_mean_4, realized_default_rate_roll_mean_4, realized_profit_roll_mean_4_scaled, realized_profit_roll_std_4_scaled, threshold_new_delta_lag_1, threshold_repeat_delta_lag_1.

| # | Feature | Layer | Type | Normalized | Definition |
| --- | --- | --- | --- | --- | --- |
| 1 | `week_progress` | A | scalar | yes | Interactive week index divided by the episode horizon. |
| 2 | `approval_rate_current` | A | scalar | yes | Last observed overall approval rate. |
| 3 | `approval_rate_new` | A | segmented | yes | Last observed approval rate for new applicants. |
| 4 | `approval_rate_repeat` | A | segmented | yes | Last observed approval rate for repeat applicants. |
| 5 | `rolling_realized_default_rate` | A | rolling | yes | Rolling realized default rate from the simulator reward window. |
| 6 | `expected_default_rate_current` | A | scalar | yes | Last observed expected default rate among accepted applications. |
| 7 | `realized_profit_scaled` | A | scalar | yes | Last observed realized profit scaled by 10,000. |
| 8 | `rolling_profit_volatility_scaled` | A | rolling | yes | Rolling realized profit volatility scaled by 10,000. |
| 9 | `projected_capital_usage_ratio` | A | scalar | yes | Projected capital usage ratio after the last action. |
| 10 | `outstanding_ratio` | A | scalar | yes | Outstanding principal divided by the capital limit. |
| 11 | `threshold_new_normalized` | A | scalar | yes | Last applied new-client threshold normalized to the configured threshold range. |
| 12 | `threshold_repeat_normalized` | A | scalar | yes | Last applied repeat-client threshold normalized to the configured threshold range. |
| 13 | `repeat_share_current` | B | segmented | yes | Share of applications coming from repeat clients in the last observed week. |
| 14 | `expected_profit_per_application_scaled` | B | scalar | yes | Last observed expected cohort profit per application scaled by 100. |
| 15 | `expected_npv_per_application_scaled` | B | scalar | yes | Last observed expected cohort NPV per application scaled by 100. |
| 16 | `realized_npv_scaled` | B | scalar | yes | Last observed realized NPV scaled by 10,000. |
| 17 | `weekly_reward_scaled` | B | scalar | yes | Last observed weekly reward scaled by 10,000. |
| 18 | `threshold_gap_normalized` | B | scalar | yes | Difference between new and repeat thresholds normalized by the threshold range. |
| 19 | `capital_headroom_ratio` | B | scalar | yes | Remaining capital headroom after the last action. |
| 20 | `realized_expected_default_gap` | B | scalar | yes | Gap between rolling realized default rate and last observed expected default rate. |
| 21 | `approval_rate_lag_2` | C | lagged | yes | Overall approval rate observed two interactive weeks ago. |
| 22 | `realized_default_rate_lag_2` | C | lagged | yes | Realized default rate observed two interactive weeks ago. |
| 23 | `realized_profit_lag_2_scaled` | C | lagged | yes | Realized profit from two interactive weeks ago scaled by 10,000. |
| 24 | `capital_usage_lag_2` | C | lagged | yes | Projected capital usage ratio observed two interactive weeks ago. |
| 25 | `approval_rate_roll_mean_4` | C | rolling | yes | Mean approval rate over the last four interactive weeks. |
| 26 | `realized_default_rate_roll_mean_4` | C | rolling | yes | Mean realized default rate over the last four interactive weeks. |
| 27 | `realized_profit_roll_mean_4_scaled` | C | rolling | yes | Mean realized profit over the last four interactive weeks scaled by 10,000. |
| 28 | `realized_profit_roll_std_4_scaled` | C | rolling | yes | Standard deviation of realized profit over the last four interactive weeks scaled by 10,000. |
| 29 | `threshold_new_delta_lag_1` | C | lagged | yes | Week-over-week change in the new-client threshold normalized by the threshold range. |
| 30 | `threshold_repeat_delta_lag_1` | C | lagged | yes | Week-over-week change in the repeat-client threshold normalized by the threshold range. |

## 50D

Baseline compatibility note: the first 12 ordered features are identical to the original 12D baseline.

Newly added at this level: approval_rate_new_roll_mean_4, approval_rate_repeat_roll_mean_4, approval_rate_new_lag_2, approval_rate_repeat_lag_2, expected_default_rate_new_current, expected_default_rate_repeat_current, expected_profit_new_per_accept_scaled, expected_profit_repeat_per_accept_scaled, accepted_new_share_current, accepted_repeat_share_current, reward_roll_mean_4_scaled, reward_roll_std_4_scaled, cumulative_reward_to_date_scaled, cumulative_profit_to_date_scaled, capital_usage_roll_std_4, outstanding_ratio_delta_lag_1, projected_minus_outstanding_gap, threshold_gap_lag_2, threshold_gap_delta_lag_1, applications_ratio_current.

| # | Feature | Layer | Type | Normalized | Definition |
| --- | --- | --- | --- | --- | --- |
| 1 | `week_progress` | A | scalar | yes | Interactive week index divided by the episode horizon. |
| 2 | `approval_rate_current` | A | scalar | yes | Last observed overall approval rate. |
| 3 | `approval_rate_new` | A | segmented | yes | Last observed approval rate for new applicants. |
| 4 | `approval_rate_repeat` | A | segmented | yes | Last observed approval rate for repeat applicants. |
| 5 | `rolling_realized_default_rate` | A | rolling | yes | Rolling realized default rate from the simulator reward window. |
| 6 | `expected_default_rate_current` | A | scalar | yes | Last observed expected default rate among accepted applications. |
| 7 | `realized_profit_scaled` | A | scalar | yes | Last observed realized profit scaled by 10,000. |
| 8 | `rolling_profit_volatility_scaled` | A | rolling | yes | Rolling realized profit volatility scaled by 10,000. |
| 9 | `projected_capital_usage_ratio` | A | scalar | yes | Projected capital usage ratio after the last action. |
| 10 | `outstanding_ratio` | A | scalar | yes | Outstanding principal divided by the capital limit. |
| 11 | `threshold_new_normalized` | A | scalar | yes | Last applied new-client threshold normalized to the configured threshold range. |
| 12 | `threshold_repeat_normalized` | A | scalar | yes | Last applied repeat-client threshold normalized to the configured threshold range. |
| 13 | `repeat_share_current` | B | segmented | yes | Share of applications coming from repeat clients in the last observed week. |
| 14 | `expected_profit_per_application_scaled` | B | scalar | yes | Last observed expected cohort profit per application scaled by 100. |
| 15 | `expected_npv_per_application_scaled` | B | scalar | yes | Last observed expected cohort NPV per application scaled by 100. |
| 16 | `realized_npv_scaled` | B | scalar | yes | Last observed realized NPV scaled by 10,000. |
| 17 | `weekly_reward_scaled` | B | scalar | yes | Last observed weekly reward scaled by 10,000. |
| 18 | `threshold_gap_normalized` | B | scalar | yes | Difference between new and repeat thresholds normalized by the threshold range. |
| 19 | `capital_headroom_ratio` | B | scalar | yes | Remaining capital headroom after the last action. |
| 20 | `realized_expected_default_gap` | B | scalar | yes | Gap between rolling realized default rate and last observed expected default rate. |
| 21 | `approval_rate_lag_2` | C | lagged | yes | Overall approval rate observed two interactive weeks ago. |
| 22 | `realized_default_rate_lag_2` | C | lagged | yes | Realized default rate observed two interactive weeks ago. |
| 23 | `realized_profit_lag_2_scaled` | C | lagged | yes | Realized profit from two interactive weeks ago scaled by 10,000. |
| 24 | `capital_usage_lag_2` | C | lagged | yes | Projected capital usage ratio observed two interactive weeks ago. |
| 25 | `approval_rate_roll_mean_4` | C | rolling | yes | Mean approval rate over the last four interactive weeks. |
| 26 | `realized_default_rate_roll_mean_4` | C | rolling | yes | Mean realized default rate over the last four interactive weeks. |
| 27 | `realized_profit_roll_mean_4_scaled` | C | rolling | yes | Mean realized profit over the last four interactive weeks scaled by 10,000. |
| 28 | `realized_profit_roll_std_4_scaled` | C | rolling | yes | Standard deviation of realized profit over the last four interactive weeks scaled by 10,000. |
| 29 | `threshold_new_delta_lag_1` | C | lagged | yes | Week-over-week change in the new-client threshold normalized by the threshold range. |
| 30 | `threshold_repeat_delta_lag_1` | C | lagged | yes | Week-over-week change in the repeat-client threshold normalized by the threshold range. |
| 31 | `approval_rate_new_roll_mean_4` | D | rolling | yes | Mean new-client approval rate over the last four interactive weeks. |
| 32 | `approval_rate_repeat_roll_mean_4` | D | rolling | yes | Mean repeat-client approval rate over the last four interactive weeks. |
| 33 | `approval_rate_new_lag_2` | D | lagged | yes | New-client approval rate observed two interactive weeks ago. |
| 34 | `approval_rate_repeat_lag_2` | D | lagged | yes | Repeat-client approval rate observed two interactive weeks ago. |
| 35 | `expected_default_rate_new_current` | D | segmented | yes | Last observed expected default rate for accepted new-client applications. |
| 36 | `expected_default_rate_repeat_current` | D | segmented | yes | Last observed expected default rate for accepted repeat-client applications. |
| 37 | `expected_profit_new_per_accept_scaled` | D | segmented | yes | Expected profit of accepted new-client loans per accepted loan, scaled by 100. |
| 38 | `expected_profit_repeat_per_accept_scaled` | D | segmented | yes | Expected profit of accepted repeat-client loans per accepted loan, scaled by 100. |
| 39 | `accepted_new_share_current` | D | segmented | yes | Share of accepted loans that came from new applicants in the last observed week. |
| 40 | `accepted_repeat_share_current` | D | segmented | yes | Share of accepted loans that came from repeat applicants in the last observed week. |
| 41 | `reward_roll_mean_4_scaled` | D | rolling | yes | Mean weekly reward over the last four interactive weeks scaled by 10,000. |
| 42 | `reward_roll_std_4_scaled` | D | rolling | yes | Standard deviation of weekly reward over the last four interactive weeks scaled by 10,000. |
| 43 | `cumulative_reward_to_date_scaled` | D | scalar | yes | Cumulative reward over interactive history scaled by 100,000. |
| 44 | `cumulative_profit_to_date_scaled` | D | scalar | yes | Cumulative realized profit over interactive history scaled by 100,000. |
| 45 | `capital_usage_roll_std_4` | D | rolling | yes | Standard deviation of projected capital usage over the last four interactive weeks. |
| 46 | `outstanding_ratio_delta_lag_1` | D | lagged | yes | Week-over-week change in the outstanding capital ratio. |
| 47 | `projected_minus_outstanding_gap` | D | scalar | yes | Gap between projected capital usage and currently outstanding capital ratio. |
| 48 | `threshold_gap_lag_2` | D | lagged | yes | Threshold gap observed two interactive weeks ago, normalized by the threshold range. |
| 49 | `threshold_gap_delta_lag_1` | D | lagged | yes | Week-over-week change in the threshold gap, normalized by the threshold range. |
| 50 | `applications_ratio_current` | D | scalar | yes | Last observed application volume divided by the profile-scaled weekly application baseline. |
