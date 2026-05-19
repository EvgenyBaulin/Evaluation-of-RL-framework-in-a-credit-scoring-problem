from pathlib import Path
import sys
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl_credit_scoring_sim.utils.device_utils import detect_device, benchmark_device, n_cpu_workers
from rl_credit_scoring_sim.evaluation.dimensionality import run_dimensionality_experiment

from rich.logging import RichHandler
from rl_credit_scoring_sim.utils.progress import console

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the controlled state-dimensionality experiment "
                    "(12D / 20D / 30D / 50D) with all 6 agents: "
                    "dqn, double_dqn, a3c, a2c, ppo, sac.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile name from configs/run_profile.yaml. Defaults to active_profile.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Compute device for neural networks. "
             "'auto' selects MPS > CUDA > CPU automatically.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Run each state dimension in a separate subprocess. "
             "Recommended when you have ≥4 CPU cores and outputs do not overlap.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a quick device benchmark before starting the experiment.",
    )
    args = parser.parse_args()

    device = detect_device() if args.device == "auto" else args.device

    logger.info("=" * 60)
    logger.info("Dimensionality Experiment")
    logger.info("  profile     : %s", args.profile or "active_profile")
    logger.info("  device      : %s", device)
    logger.info("  parallelize : %s", args.parallelize)
    logger.info("  cpu workers : %d", n_cpu_workers())
    logger.info("=" * 60)

    if args.benchmark:
        logger.info("Running device benchmark …")
        ok = benchmark_device(device)
        logger.info("Benchmark: %s", "PASS" if ok else "FAIL (falling back to cpu)")
        if not ok:
            device = "cpu"

    run_dimensionality_experiment(
        project_root=PROJECT_ROOT,
        profile=args.profile,
        device=device,
        parallelize=args.parallelize,
    )
