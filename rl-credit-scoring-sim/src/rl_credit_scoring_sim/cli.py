from __future__ import annotations

import argparse
from pathlib import Path

from rl_credit_scoring_sim.evaluation.dimensionality import run_dimensionality_experiment
from rl_credit_scoring_sim.evaluation.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RL credit scoring simulation pipeline.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile name from configs/run_profile.yaml. Defaults to active_profile.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        choices=[12, 20, 30, 50],
        help="Override the observation dimensionality for a single pipeline run.",
    )
    parser.add_argument(
        "--dimensionality-experiment",
        action="store_true",
        help="Run the controlled 12/20/30/50 state-dimensionality comparison.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device for neural networks. 'auto' detects MPS > CUDA > CPU.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Run each state dimension in a separate process (dimensionality experiment only).",
    )
    return parser


def main() -> None:
    from rl_credit_scoring_sim.utils.device_utils import detect_device
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    device = detect_device() if args.device == "auto" else args.device
    if args.dimensionality_experiment:
        run_dimensionality_experiment(
            project_root=project_root,
            profile=args.profile,
            device=device,
            parallelize=getattr(args, "parallelize", False),
        )
        return
    if args.state_dim is not None:
        from rl_credit_scoring_sim.config import load_run_config, load_scenarios
        from rl_credit_scoring_sim.evaluation.pipeline import execute_pipeline

        config = load_run_config(
            project_root,
            profile=args.profile,
            overrides={"state_dim": args.state_dim, "device": device},
        )
        scenarios = load_scenarios(project_root)
        execute_pipeline(config=config, scenarios=scenarios)
        return
    run_pipeline(project_root=project_root, profile=args.profile)


if __name__ == "__main__":
    main()
