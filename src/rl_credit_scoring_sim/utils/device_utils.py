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


def resolve_device(requested: str = "auto", validate: bool = True) -> str:
    """Resolve a device request to a concrete, working device string.

    This is the single entry point that the CLI and experiment runners should
    use so device selection is automatic and robust:

    - ``"auto"`` picks the best available accelerator (MPS > CUDA > CPU).
    - An explicit ``"cuda"``/``"mps"``/``"cpu"`` is honored as-is.
    - When ``validate`` is true and the resolved device is an accelerator, a
      quick smoke test runs; if it fails (e.g. a broken CUDA driver or a
      requested accelerator that is not really present) the device is silently
      downgraded to CPU so a long unattended run never crashes on startup.

    Args:
        requested: "auto" | "cpu" | "mps" | "cuda"
        validate: run a matmul smoke test on accelerators before committing.

    Returns:
        A concrete device string: "cpu" | "mps" | "cuda".
    """
    device = detect_device() if requested == "auto" else requested
    if validate and device != "cpu":
        if not benchmark_device(device):
            logger.warning(
                "Requested device '%s' is not functional — falling back to CPU.",
                device,
            )
            return "cpu"
    return device


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
