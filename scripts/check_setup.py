"""
Pre-flight verification: agents, device, config, dependencies.
Run before starting experiments to catch problems early.

Usage:
    python scripts/check_setup.py
"""
from pathlib import Path
import sys
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REQUIRED_AGENTS = ["dqn", "double_dqn", "a3c", "a2c", "ppo", "sac"]
REQUIRED_PACKAGES = ["numpy", "pandas", "torch", "gymnasium", "stable_baselines3", "rich", "tqdm"]


def check_dependencies() -> bool:
    logger.info("\n── Dependencies ──")
    ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            logger.info("  ✓ %s", pkg)
        except ImportError:
            logger.error("  ✗ %s  (not installed)", pkg)
            ok = False
    return ok


def check_agents() -> bool:
    logger.info("\n── Agents ──")
    try:
        from rl_credit_scoring_sim.agents.factory import make_agent, controller_mode
    except Exception as exc:
        logger.error("  ✗ factory import failed: %s", exc)
        return False

    import numpy as np

    class _DummyConfig(dict):
        pass

    ok = True
    for name in REQUIRED_AGENTS:
        try:
            # Minimal config so make_agent doesn't crash on missing keys
            cfg = {
                "state_dim": 12,
                "device": "cpu",
                "environment": {"horizon_weeks": 4},
                "agents": {
                    name: {
                        "hidden_sizes": [32],
                        "learning_rate": 1e-3,
                        "gamma": 0.99,
                        "batch_size": 8,
                        "replay_size": 100,
                        "learning_starts": 10,
                        "train_frequency": 1,
                        "gradient_steps": 1,
                        "target_update_interval": 10,
                        "epsilon_start": 1.0,
                        "epsilon_end": 0.05,
                        "epsilon_decay_fraction": 0.5,
                        "entropy_coef": 0.01,
                        "value_coef": 0.5,
                        "n_steps": 16,
                        "gae_lambda": 0.95,
                        "ent_coef": 0.01,
                        "clip_range": 0.2,
                        "buffer_size": 100,
                        "learning_starts": 10,
                        "tau": 0.02,
                        "train_freq": 1,
                        "policy_kwargs": {"net_arch": [32]},
                    }
                },
            }
            mode = controller_mode(name)
            action_dim = 11  # threshold grid size
            agent = make_agent(name=name, config=cfg, action_dim=action_dim, obs_dim=12)
            logger.info("  ✓ %s  (mode=%s)", name, mode)
        except Exception as exc:
            logger.error("  ✗ %s  — %s", name, exc)
            ok = False
    return ok


def check_device() -> bool:
    logger.info("\n── Device ──")
    try:
        from rl_credit_scoring_sim.utils.device_utils import detect_device, benchmark_device, n_cpu_workers
        device = detect_device()
        workers = n_cpu_workers()
        logger.info("  primary device : %s", device)
        logger.info("  cpu workers    : %d", workers)
        ok = benchmark_device(device)
        logger.info("  benchmark      : %s", "PASS" if ok else "FAIL")
        return ok
    except Exception as exc:
        logger.error("  ✗ device check failed: %s", exc)
        return False


def check_config() -> bool:
    logger.info("\n── Configuration ──")
    try:
        from rl_credit_scoring_sim.config import load_run_config, load_scenarios
        for profile in ("quick", "full"):
            cfg = load_run_config(PROJECT_ROOT, profile=profile)
            agents = cfg["controllers"]["agents"]
            logger.info("  ✓ profile=%s  agents=%s", profile, agents)
        load_scenarios(PROJECT_ROOT)
        logger.info("  ✓ scenarios.yaml loaded")
        return True
    except Exception as exc:
        logger.error("  ✗ config error: %s", exc)
        return False


def check_outputs_dir() -> bool:
    logger.info("\n── Output directories ──")
    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    logger.info("  ✓ outputs/ ready at %s", outputs)
    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Project Setup Verification")
    logger.info("=" * 60)

    checks = [
        ("Dependencies", check_dependencies),
        ("Agents (6)", check_agents),
        ("Device / MPS", check_device),
        ("Configuration", check_config),
        ("Output dirs", check_outputs_dir),
    ]

    results = {}
    for label, fn in checks:
        try:
            results[label] = fn()
        except Exception as exc:
            logger.error("  ✗ %s check crashed: %s", label, exc)
            results[label] = False

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for label, passed in results.items():
        logger.info("  %s  %s", "✓ PASS" if passed else "✗ FAIL", label)

    all_ok = all(results.values())
    logger.info("")
    if all_ok:
        logger.info("✓ All checks passed. Ready to run experiments.")
        logger.info("")
        logger.info("  Quick smoke:  python scripts/run_pipeline.py --state-dim 12 --profile quick")
        logger.info("  Full run:     python scripts/run_dimensionality_experiment.py --profile quick --device auto")
    else:
        logger.error("✗ Some checks failed. Fix the issues above before running experiments.")

    logger.info("=" * 60)
    sys.exit(0 if all_ok else 1)
