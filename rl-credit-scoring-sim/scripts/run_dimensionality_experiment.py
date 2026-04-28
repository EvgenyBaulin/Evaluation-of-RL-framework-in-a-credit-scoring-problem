from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_credit_scoring_sim.evaluation.dimensionality import run_dimensionality_experiment


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the controlled state-dimensionality experiment.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile name from configs/run_profile.yaml. Defaults to active_profile.",
    )
    args = parser.parse_args()
    run_dimensionality_experiment(
        project_root=PROJECT_ROOT, profile=args.profile)
