from __future__ import annotations

from pathlib import Path


def ensure_directories(config: dict) -> dict[str, Path]:
    root = Path(config["project_root"])
    paths = {}
    for key, relative in config["paths"].items():
        path = root / relative
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths
