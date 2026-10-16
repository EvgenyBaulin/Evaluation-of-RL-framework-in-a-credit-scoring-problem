# Reinforcement Learning vs Rule-Based Policies for Dynamic Credit Threshold Control

Code, results and manuscript for the article on weekly acceptance-threshold control in credit scoring with reinforcement learning.

Authors: E. Baulin, P. P. Lukianchenko.

## What this repository studies

A controller sets two acceptance thresholds every week, one for new clients and one for repeat clients. Accepted loans produce delayed repayments, defaults, recoveries, capital usage and profit over later weeks. The article asks:

> Under which conditions does reinforcement learning provide a robust improvement over rule-based threshold policies in a delayed-outcome credit-scoring environment?

A controlled experiment varies only the observation size (`12`, `20`, `30`, `50` features) and keeps everything else fixed: simulator dynamics, reward, scenarios, seeds, train/eval protocol and baselines.

- RL agents: DQN, Double-DQN, A3C, A2C, PPO, SAC
- Rule-based baselines: static threshold, split-policy static, profit-oriented, risk-aware weekly, constraint-aware weekly
- Scenarios: base market, drift, adverse stress, class-imbalance shift, noise, split-policy dynamics

## Repository layout

```text
paper/        manuscript: article.tex, article.pdf, references.bib, figures/
src/          simulator, environment, agents, baselines, evaluation (package rl_credit_scoring_sim)
scripts/      entry points: experiment runner, statistical tests, figure generation
configs/      run_profile.yaml (quick / full / cuda profiles), scenarios.yaml
outputs/      summary tables and figures used in the article
docs/         state-feature manifest and experiment report
```

Key files in `outputs/`:

| File | Content |
| --- | --- |
| `dimension_comparison_summary.csv` | all controllers, all dimensions |
| `best_rl_by_dimension.csv` | best RL agent per state dimension |
| `overall_best_by_dimension.csv` | best controller overall per dimension |
| `statistical_hypothesis_tests.csv` | paired tests for hypotheses H1-H3 |
| `scenario_winner_counts.csv` | per-scenario wins, RL vs baselines |
| `metric_vs_dimension_*.png` | metrics against state dimension |
| `lc_summary.json` | Lending Club aggregates used to calibrate the simulator |

Per-dimension run folders (`outputs/dim_*/`) are large and not tracked; the scripts regenerate them.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py
```

## Reproducing the results

```bash
# smoke test, one state size
python scripts/run_pipeline.py --state-dim 20 --profile quick

# controlled dimensionality experiment
python scripts/run_dimensionality_experiment.py --profile quick
python scripts/run_dimensionality_experiment.py --profile full --device auto --parallelize

# statistical tests on the produced outputs
python scripts/run_hypothesis_tests.py
```

Profiles are defined in `configs/run_profile.yaml`. The `full` profile (8 seeds, 1500 training episodes, 52-week horizon) took about 9 hours on an Apple M3 Pro with `--parallelize`.

Optional scripts:

- `scripts/generate_learning_curves.py`: training diagnostics from `outputs/dim_*/training_logs.csv`
- `scripts/regenerate_saturation_figure.py`: rebuilds `paper/figures/dimensionality_saturation.png` from the summary CSVs
- `scripts/real_data_comparison.py`: calibration figure from the Lending Club data (place the CSV in `data/`)
- `scripts/generate_simulator_diagrams.py`: simulator schematics

## Building the manuscript

```bash
cd paper
pdflatex -interaction=nonstopmode article.tex
pdflatex -interaction=nonstopmode article.tex
```

The bibliography is inline, so no BibTeX pass is needed. The `elsarticle` class is required.

## Feature sets

The exact composition of the 12D, 20D, 30D and 50D observations is listed in [`docs/state_dimension_manifest.md`](docs/state_dimension_manifest.md). The first 12 features are identical in every configuration.
