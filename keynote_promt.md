**Deck Strategy**
Recommend `14` slides. The strongest pre-defense version is not “RL beats everything,” but “this thesis formulates credit-scoring as weekly portfolio threshold control with delayed outcomes, implements that logic in a synthetic simulator, and then runs a controlled state-dimensionality experiment showing that observation design materially changes RL quality.”

Lead with problem framing and simulator logic before results. The committee will likely accept the empirical message if you are explicit about three things early: the environment is synthetic, the control object is weekly thresholds rather than pointwise approvals, and the main overall winner in the current quick-profile outputs is still a rule-based baseline.

**Slide-by-Slide Plan**

**Slide 1**  
Title: `Weekly Threshold Control in Credit Scoring`  
Purpose: Open with the exact topic and signal that this is a methodological research study, not a product demo.  
Slide bullets:

- Pre-defense: RL-based weekly threshold control for a synthetic credit portfolio
- Focus of the current repository: controlled state-dimensionality experiment
- Core question: how much does observation design change controller quality when everything else is fixed?
  Exact repository visuals to include:
- None.  
  Short speaker note: Say immediately that the study is about weekly portfolio control in a simulator with delayed outcomes, not about static PD classification on a public dataset.

**Slide 2**  
Title: `Problem, Importance, and Research Question`  
Purpose: Motivate why the problem matters and state the thesis question in one slide.  
Slide bullets:

- Real lending policy is often revised in batches, not one application at a time
- The decision is not only “approve or reject now,” but “which weekly cutoff keeps the portfolio profitable, solvent, and stable?”
- Research question: when only `state_dim` changes, how much does RL quality change?
- Hypothesis: richer portfolio state should help up to a point, then saturate
  Exact repository visuals to include:
- None.  
  Short speaker note: Emphasize that the thesis contribution is mainly in problem formulation plus controlled evaluation, not in proposing a new RL algorithm.

**Slide 3**  
Title: `Why This Is a Weekly Portfolio-Control Problem`  
Purpose: Defend the weekly-threshold framing against the obvious “why not classification?” question.  
Slide bullets:

- The agent controls weekly thresholds, not individual approvals
- Action is a threshold pair: \((\tau^{new}\_t,\tau^{repeat}\_t)\)
- Accepted loans affect future defaults, recoveries, capital usage, and portfolio composition
- This makes the task sequential, portfolio-oriented, and economically interpretable
  Exact repository visuals to include:
- None. Optional schematic derived from [threshold_env.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/env/threshold_env.py) and [thresholds.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/utils/thresholds.py).  
  Short speaker note: Say that this slide answers “what exactly is being controlled?” before you show any RL results.

**Slide 4**  
Title: `Simulator Logic and Episode Timeline`  
Purpose: Show how one episode works and why the simulator is a valid research object.  
Slide bullets:

- `8` warm-up weeks with default thresholds populate the portfolio state
- Each interactive week: scenario state -> applications -> accept above thresholds -> schedule future events
- Interactive horizon in the quick profile: `26` weeks
- Terminal settlement settles pending interactive cash flows after the last decision week
  Exact repository visuals to include:
- None. Build a simple timeline from [simulator.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/env/simulator.py) and [run_profile.yaml](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/configs/run_profile.yaml).  
  Short speaker note: Stress that terminal settlement is not cosmetic; it prevents end-of-horizon truncation bias.

**Slide 5**  
Title: `Delayed Outcomes, Delayed Reward, and Terminal Settlement`  
Purpose: Explain why this is not a myopic one-step optimization problem.  
Slide bullets:

- Accepted loans pay, default, or recover after future delays
- `delayed_reward: true` means reward uses realized profit and realized NPV, not only current-cohort expectation
- Reward realization and outcome realization are both delayed in the active design
- Terminal settlement avoids undercounting late-originated loans near the horizon end
  Exact repository visuals to include:
- None on the main slide; reserve the diagnostic evidence for Slide 12. Source files: [reward.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/env/reward.py) and [simulator.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/env/simulator.py).  
  Short speaker note: This slide should answer “why can’t we evaluate policies from current approvals alone?”

**Slide 6**  
Title: `Split Policy, Reward, and Controller Set`  
Purpose: Defend three core implementation choices together: separate segments, multi-objective reward, and the comparison set.  
Slide bullets:

- New and repeat clients are modeled separately: different score, default, recovery, size, and duration logic
- `split_policy: true`; thresholds are separate for new and repeat clients
- Reward is multi-objective, not pure profit only:
- \(r*t=\pi_t+0.25\,NPV_t-0.35\cdot100\hat d_t-P*{default}-P*{approval}-P*{capital}-P\_{volatility}\)
- Penalty targets: default `0.12`, approval `0.42`, capital usage `0.82`, volatility `8500`
- Controller set: `DQN`, `Double-DQN`, `PPO`, `SAC` plus `5` baselines; DQN/DDQN are discrete, PPO/SAC continuous
  Exact repository visuals to include:
- None. Source files: [reward.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/env/reward.py), [policies.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/baselines/policies.py), [scenarios.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/env/scenarios.py), [factory.py](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/src/rl_credit_scoring_sim/agents/factory.py).  
  Short speaker note: This slide answers “why separate new and repeat?” and “what exactly is optimized?”

**Slide 7**  
Title: `Controlled Experimental Protocol`  
Purpose: Make the methodological control explicit before showing the dimension results.  
Slide bullets:

- Active profile: `quick`
- `3` seeds, `12` training episodes, `4` evaluation runs, `6` scenarios
- `72` evaluation runs per controller per dimension
- Threshold range `35..85` with step `5`; split discrete grid gives `121` threshold pairs
- Only `state_dim` changes; same scenarios, reward, warm-up, terminal settlement, controllers, and metrics
  Exact repository visuals to include:
- Retype a compact protocol box from [run_profile.yaml](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/configs/run_profile.yaml) and [scenarios.yaml](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/configs/scenarios.yaml).  
  Short speaker note: State clearly that the dimension experiment is controlled because the first 12 features and all environment mechanics are held fixed.

**Slide 8**  
Title: `Observation Design and the State-Dimensionality Experiment`  
Purpose: Explain what 12D, 20D, 30D, and 50D actually mean.  
Slide bullets:

- `12D`: compact weekly dashboard
- `20D`: adds immediate economic context and reward-related state
- `30D`: adds lagged and rolling memory
- `50D`: adds richer segment detail, cumulative state, and stability descriptors
- The first `12` ordered features remain unchanged in every compared state
  Exact repository visuals to include:
- Build a 4-row summary table from [state_dimension_manifest.md](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/state_dimension_manifest.md).  
  Short speaker note: Frame dimension as controlled information increase, not as a different simulator or different task.

**Slide 9**  
Title: `Headline Results: Best Overall vs Best RL`  
Purpose: Establish the main empirical fact pattern honestly.  
Slide bullets:

- Best overall controller in all four dimensions: `profit_oriented` baseline
- Overall best values stay the same across dimensions because the rule-based baselines do not use `state_dim`
- Best RL frontier improves from `12D` to `30D`, then plateaus
- Reward gap to best overall shrinks from about `20.8k` at `12D` to about `9.9k` at `30D/50D`
  Exact repository visuals to include:
- [metric_vs_dimension_cumulative_reward.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/metric_vs_dimension_cumulative_reward.png)
- Build a small slide table from [best_rl_by_dimension.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/best_rl_by_dimension.csv) and [overall_best_by_dimension.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/overall_best_by_dimension.csv).  
  Short speaker note: Say explicitly that the results do not support a blanket RL-superiority claim.

**Slide 10**  
Title: `What the Dimensionality Experiment Shows`  
Purpose: Interpret the cross-dimension effect precisely.  
Slide bullets:

- Best RL by dimension: `12 DQN`, `20 Double-DQN`, `30 DQN`, `50 DQN`
- Largest improvement is `20 -> 30`: `+5661.8` profit, `+8407.7` reward, lower default, lower approval, lower capital usage
- `30 -> 50` is saturation: only `+188.5` profit, `-21.3` reward, slightly worse default and selectivity
- Algorithm effect is non-uniform: DQN improves, Double-DQN peaks earlier and collapses at `50D`, PPO is unchanged, SAC remains unstable
- Most balanced RL state size in the quick-profile outputs: `30D`
  Exact repository visuals to include:
- [metric_vs_dimension_expected_profit.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/metric_vs_dimension_expected_profit.png)
- [metric_vs_dimension_default_rate.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/metric_vs_dimension_default_rate.png)
- Support with [dimension_comparison_summary.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dimension_comparison_summary.csv).  
  Short speaker note: The defensible claim is not “50D is worse,” but “50D adds almost no extra benefit beyond 30D while weakening the balance.”

**Slide 11**  
Title: `Scenario Heterogeneity and Metric Trade-Offs`  
Purpose: Show that controller rankings change across scenarios and across objectives.  
Slide bullets:

- `base_market`: RL is strongest; in `50D`, DQN is best overall by reward, profit, and default rate
- `split_policy_dynamics`: richer segmented state matters; `50D` DQN nearly matches the best baseline and has the lowest default rate
- `adverse_stress` and `drift`: baselines dominate
- `noise`: reward winner and profit winner differ; reward best = `profit_oriented`, profit best = `risk_aware_weekly`
- `class_imbalance_shift`: `constraint_aware_weekly` wins overall, while DQN gives the lowest default in `30D/50D`
  Exact repository visuals to include:
- Main slide source: [dim_50/main_scenario_summary.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_50/main_scenario_summary.csv)
- Supporting raw figure if you want one repository chart: [expected_profit_by_scenario_dim50.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_50/expected_profit_by_scenario_dim50.png)
- Optional risk companion: [default_rate_by_scenario_dim50.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_50/default_rate_by_scenario_dim50.png)  
  Short speaker note: The full per-scenario bar charts are honest but dense; for the committee, a retyped winner table is usually clearer than showing all controller bars.

**Slide 12**  
Title: `Diagnostics: Early Loss, Late Gain and Threshold Behavior`  
Purpose: Use the repository diagnostics to explain delayed evaluation and the current limit of RL adaptivity.  
Slide bullets:

- Early and final rankings can differ because outcomes are delayed
- In `30D adverse_stress`, DQN has early gap `-643.46` vs `static_threshold`, but final gap `+250,578.20`
- Best RL controllers in the quick profile are episode-static: `72/72` constant-threshold runs for each best RL controller
- Current evidence supports better threshold-pair selection more strongly than within-episode weekly adaptation
  Exact repository visuals to include:
- [dim_30/locally_worse_globally_better.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_30/locally_worse_globally_better.png)
- [dim_30/locally_worse_globally_better.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_30/locally_worse_globally_better.csv)
- [dim_30/best_rl_threshold_paths_dim30.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_30/best_rl_threshold_paths_dim30.png)  
  Short speaker note: Volunteer this limitation yourself; it makes the rest of the deck more credible.

**Slide 13**  
Title: `Novelty and Limitations`  
Purpose: State the contribution honestly and bound it.  
Slide bullets:

- Novelty: weekly portfolio threshold control, split new/repeat policy, delayed outcomes, delayed reward, terminal settlement, controlled state-dimension study
- Strong baselines are part of the contribution, not a nuisance
- Limitations: synthetic environment, short quick-profile training budget, reward-shaped objective, scenario-family external validity only
- Important reporting nuance: exported `expected_profit_mean` is a legacy label; current code sums realized episode profit plus terminal settlement
  Exact repository visuals to include:
- None.  
  Short speaker note: Say the contribution is mainly in problem formulation, simulator design, and controlled experimental methodology.

**Slide 14**  
Title: `Final Conclusion and Next Steps`  
Purpose: Close with the strongest defensible claim and the natural extension path.  
Slide bullets:

- The project studies weekly threshold control for a synthetic credit portfolio with delayed outcomes
- Observation design materially changes RL quality, but not uniformly across algorithms
- In the quick-profile outputs, `30D` is the most balanced RL state size; the strongest overall controller is still a rule-based baseline
- Future work: full-profile runs, stronger training budgets, truly adaptive threshold paths, reward-sensitivity checks, calibration to real or semi-synthetic lending data
  Exact repository visuals to include:
- Optional closing table from [best_rl_by_dimension.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/best_rl_by_dimension.csv).  
  Short speaker note: End on “state design matters” rather than “RL wins.”

**Figures and Tables to Use**

- [metric_vs_dimension_cumulative_reward.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/metric_vs_dimension_cumulative_reward.png) is the strongest headline figure because it already compares `Best RL` and `Best Overall` with confidence bands.
- [metric_vs_dimension_expected_profit.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/metric_vs_dimension_expected_profit.png) plus [metric_vs_dimension_default_rate.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/metric_vs_dimension_default_rate.png) is the best pair for defending “30D is most balanced.”
- [best_rl_by_dimension.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/best_rl_by_dimension.csv) and [overall_best_by_dimension.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/overall_best_by_dimension.csv) should become one compact summary table in the deck.
- [dim_50/main_scenario_summary.csv](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_50/main_scenario_summary.csv) is a better main-slide source than the raw scenario bar charts, because the bar charts include all controllers and can be visually crowded.
- [expected_profit_by_scenario_dim50.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_50/expected_profit_by_scenario_dim50.png) and [default_rate_by_scenario_dim50.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_50/default_rate_by_scenario_dim50.png) are good appendix figures or side-support visuals, not the clearest main scenario slide.
- [dim_30/locally_worse_globally_better.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_30/locally_worse_globally_better.png) is the strongest figure for defending delayed evaluation.
- [dim_30/best_rl_threshold_paths_dim30.png](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/outputs/dim_30/best_rl_threshold_paths_dim30.png) is useful only if you explicitly discuss the episode-static behavior of the quick-profile best RL policies.
- [state_dimension_manifest.md](/Users/evgenybaulin/Library/Mobile Documents/com~apple~CloudDocs/Education/HSE/Data Science and Business Analytics/2025-2026 academic year/Diploma/rl-credit-scoring-sim/state_dimension_manifest.md) is the best source for the state-design slide; do not improvise feature descriptions from memory.

**Speaker Notes**

- Spend the most time on Slides `4-12`. That is where the defense is won.
- Use these recurring phrases: `synthetic`, `weekly`, `portfolio`, `delayed`, `split-policy`, `controlled experiment`, `not overall RL dominance`.
- When you first mention profit results, clarify orally that the exported column name `expected_profit_mean` is legacy naming; the current code aggregates realized episode profit plus terminal settlement.
- Do not wait for the committee to notice the zero threshold volatility. State it yourself on Slide 12 and frame it as an honest empirical boundary of the quick profile.
- If time is short, compress Slides `6-8`; do not cut Slides `5`, `9`, `10`, `11`, or `12`.

**Likely Questions and Answers**

- `Why study weekly thresholds instead of default classification?` Because the repository controls batch cutoffs for a portfolio, and the consequences arrive over future weeks through defaults, recoveries, and capital usage. Slides `3-5`.
- `Is this based on real bank data?` No. It is a synthetic, scenario-driven simulator designed for controlled policy research, not a confidential production dataset. Slides `2`, `4`, `13`.
- `What exactly changes in the dimensionality experiment?` Only `state_dim`; the first `12` ordered features stay unchanged, and the same reward, scenarios, seeds, baselines, algorithms, warm-up, horizon, and settlement logic are reused. Slides `7-8`.
- `Why are the best overall baseline rows identical across dimensions?` Because the rule-based baselines do not depend on the learned observation vector, so changing `state_dim` does not change their behavior. Slides `6`, `9`.
- `Why claim 30D is best if 50D has slightly higher profit?` Because `30D` has the highest best-RL cumulative reward, the lowest best-RL default rate, and lower approval and capital usage; `50D` adds only about `188.5` profit while slightly worsening the balance. Slide `10`.
- `If RL does not win overall, what is the contribution?` The contribution is not an RL victory claim; it is the formulation of weekly credit policy as delayed portfolio control, plus a controlled methodology showing that state design materially changes RL quality. Slides `10`, `13`, `14`.
- `Why do reward and profit winners differ?` Because reward is multi-objective and includes NPV and penalties for default, approval pressure, capital usage, and volatility. The `noise` scenario is the clean example. Slides `6`, `11`.
- `Does delayed reward really matter empirically?` Yes. The repository exports “locally worse, globally better” cases; for example, `30D DQN` in `adverse_stress` has a negative early gap but a large positive final gap versus `static_threshold`. Slides `5`, `12`.
- `Are the learned policies truly adaptive week to week?` In the quick profile, the best RL controllers are episode-static in evaluation, so the current evidence is stronger for better threshold-pair selection than for dynamic within-episode adjustment. Slide `12`.
- `Why is the CSV column called expected profit if you say realized profit?` Because the export keeps a legacy column name, but the current `episode_summary()` logic sums realized weekly profits and adds terminal settlement. Slide `13`.

**Final Presentation Narrative**
Core story: This thesis studies credit-scoring as a weekly portfolio-control problem rather than a one-shot classification problem. In the implemented simulator, the controller sets separate weekly thresholds for new and repeat clients, while accepted loans generate delayed repayments, defaults, and late recoveries. Because outcomes and reward realization are delayed, policy quality must be judged on full portfolio trajectories, and the environment includes warm-up and terminal settlement to avoid truncation bias. Within that fixed setting, the experiment changes only observation size: `12`, `20`, `30`, and `50` features, with the first `12` features held constant. The outputs show that richer state helps RL up to a point, but the effect is strongly algorithm-dependent. The best RL frontier improves materially from `12D` to `30D` and then largely saturates at `50D`, while the strongest overall controller in the quick-profile exports remains a rule-based baseline. The defensible thesis claim is therefore that state design is a first-order driver of RL quality in weekly credit-policy control.

Main empirical message: richer state improves the best RL frontier, especially from `20D` to `30D`, but the strongest overall controller in the current exports is still `profit_oriented`, and scenario heterogeneity remains substantial.

Main methodological message: the dimensionality experiment is credible because only `state_dim` changes; environment logic, delayed reward, terminal settlement, scenarios, seeds, baselines, and evaluation protocol are held fixed.

Main novelty claim: the work combines weekly portfolio threshold control, delayed cash-flow simulation, split new/repeat policy control, multi-objective reward design, and a controlled state-dimensionality study in one coherent research framework.

Most defensible final conclusion: in this repository’s quick-profile outputs, `30D` is the most balanced RL state size, but RL should not be presented as universally superior; the real finding is that observation design materially shapes RL performance in delayed, portfolio-level credit-threshold control.
