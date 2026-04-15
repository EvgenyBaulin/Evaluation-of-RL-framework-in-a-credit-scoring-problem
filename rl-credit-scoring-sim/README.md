# RL Credit Scoring Sim

Simulation-only thesis repository for weekly acceptance-threshold control in credit scoring.

## Repository focus

The main experiment is a controlled observation-dimensionality comparison for the existing weekly threshold-control RL framework.

- Supported observation sizes: `12`, `20`, `30`, `50`
- Controlled factor: only `state_dim`
- Held constant across dimensions: environment dynamics, weekly interaction logic, action semantics, threshold range, reward, delayed reward logic, delayed outcomes, warm-up, terminal settlement, seeds, scenarios, train/eval protocol, bootstrap settings, controllers, baselines, and metrics
- Baseline compatibility rule: the first 12 ordered features are unchanged in every state definition

## What the project does

- builds a synthetic credit portfolio simulator with delayed outcomes
- controls weekly thresholds rather than per-application approve/reject decisions
- supports separate thresholds for new and repeat clients
- compares static and rule-based baselines against RL agents
- evaluates DQN, Double-DQN, PPO, and SAC under identical scenarios and seeds
- exports per-dimension CSV summaries, weekly logs, plots, checkpoints, and writer handoff notes
- exports cross-dimension comparison tables and figures for thesis-style interpretation

## Reference boundary

`Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning-master` is reference-only and is not modified by this project.

## Setup

```bash
cd rl-credit-scoring-sim
python -m pip install -r requirements.txt
```

## Main commands

Run the full controlled dimensionality experiment:

```bash
python scripts/run_dimensionality_experiment.py --profile quick
```

Equivalent CLI entrypoint:

```bash
python scripts/run_pipeline.py --dimensionality-experiment --profile quick
```

Run one state size only:

```bash
python scripts/run_pipeline.py --state-dim 20 --profile quick
```

Run the full research profile:

```bash
python scripts/run_dimensionality_experiment.py --profile full
```

## Observation design

- `12D`: unchanged baseline state
- `20D`: baseline 12D + 8 extra features
- `30D`: 20D + 10 extra features
- `50D`: 30D + 20 extra features

The ordered feature registry, definitions, feature types, normalization flags, and cumulative layer structure are documented in `state_dimension_manifest.md`.

## Output structure

The dimensionality experiment writes one folder per state size:

- `outputs/dim_12/`
- `outputs/dim_20/`
- `outputs/dim_30/`
- `outputs/dim_50/`

Each dimension folder contains:

- `main_scenario_summary.csv`
- `main_overall_summary.csv`
- `run_level_metrics.csv`
- `weekly_run_metrics.csv`
- `weekly_reward_curves.csv`
- `weekly_profit_curves.csv`
- `expected_profit_by_scenario_dimXX.png`
- `cumulative_reward_curves_dimXX.png`
- `cumulative_profit_curves_dimXX.png`
- `default_rate_by_scenario_dimXX.png`
- `approval_rate_by_scenario_dimXX.png`
- `capital_usage_by_scenario_dimXX.png`
- `threshold_volatility_dimXX.png`
- `best_rl_threshold_paths_dimXX.png`

Cross-dimension outputs are written to `outputs/`:

- `dimension_comparison_summary.csv`
- `best_rl_by_dimension.csv`
- `overall_best_by_dimension.csv`
- `metric_vs_dimension_expected_profit.png`
- `metric_vs_dimension_npv.png`
- `metric_vs_dimension_cumulative_reward.png`
- `metric_vs_dimension_default_rate.png`
- `metric_vs_dimension_stability_index.png`

## Core documentation

- `state_dimension_manifest.md`: exact feature composition for 12D / 20D / 30D / 50D
- `dimensionality_experiment_report.md`: controlled-experiment summary and conclusion
- `notes/writer_handoff.md`: figure map, table map, run instructions, and writing guidance
- `notes/reference_study.md`: boundary between the original reference project and this implementation

## Project layout

```text
rl-credit-scoring-sim/
├── configs/
├── notes/
├── outputs/
├── scripts/
├── src/rl_credit_scoring_sim/
├── state_dimension_manifest.md
└── dimensionality_experiment_report.md
```

`artifacts/` remains available for legacy single-run outputs, but the controlled dimensionality experiment uses `outputs/` as the primary location.
