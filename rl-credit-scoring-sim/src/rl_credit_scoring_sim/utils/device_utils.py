"""
Compute device detection and selection.
Supports CPU, Metal Performance Shaders (MPS on Apple Silicon), and CUDA.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Literal

logger = logging.getLogger(__name__)


def detect_device() -> Literal["cpu", "mps", "cuda"]:
    """Return the best available compute device."""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.debug("MPS (Apple Silicon GPU) detected")
            return "mps"
        if torch.cuda.is_available():
            logger.debug("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
            return "cuda"
    except ImportError:
        pass
    logger.debug("No GPU detected — using CPU")
    return "cpu"


def get_torch_device(device: str = "auto"):
    """Return a torch.device for the given device string.

    Args:
        device: "auto" | "cpu" | "mps" | "cuda"
    """
    import torch
    if device == "auto":
        device = detect_device()
    return torch.device(device)


def n_cpu_workers() -> int:
    """Recommended number of parallel workers (one core reserved for OS)."""
    return max(1, mp.cpu_count() - 1)


def benchmark_device(device: str = "auto") -> bool:
    """Quick matmul smoke-test to confirm the device is functional."""
    try:
        import torch
        dev = get_torch_device(device)
        x = torch.randn(512, 512, device=dev)
        y = torch.randn(512, 512, device=dev)
        _ = torch.matmul(x, y)
        logger.info("Device benchmark passed on %s", dev)
        return True
    except Exception as exc:
        logger.warning("Device benchmark failed on %s: %s", device, exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d = detect_device()
    print(f"Primary device : {d}")
    print(f"CPU workers    : {n_cpu_workers()}")
    benchmark_device(d)
