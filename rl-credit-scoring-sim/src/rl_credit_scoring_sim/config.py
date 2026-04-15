from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ALLOWED_STATE_DIMS = {12, 20, 30, 50}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_run_config(
    root_dir: str | Path,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    raw = load_yaml(root_dir / "configs" / "run_profile.yaml")
    selected_profile = profile or raw["active_profile"]
    profile_settings = raw["profiles"][selected_profile]
    config = _deep_merge(raw["shared"], profile_settings)
    if overrides:
        config = _deep_merge(config, overrides)
    for key in ("horizon_weeks", "applications_per_week", "test_subset_fraction"):
        if key in config:
            config["environment"][key] = config[key]
    for key in ("full_ci", "bootstrap_resamples"):
        if key in config:
            config["confidence_intervals"][key] = config[key]
    config.setdefault("plotting", {})
    config["plotting"]["build_minimal_plots_only"] = config.get("build_minimal_plots_only", False)
    seed_count_key = "quick_test_seed_count" if selected_profile == "quick" else "final_comparison_seed_count"
    if seed_count_key in config:
        config["seeds"] = list(config["seeds"][: config[seed_count_key]])
    config["state_dim"] = int(config.get("state_dim", 12))
    if config["state_dim"] not in ALLOWED_STATE_DIMS:
        raise ValueError(
            f"Unsupported state_dim={config['state_dim']}. Allowed values: {sorted(ALLOWED_STATE_DIMS)}."
        )
    config["profile_name"] = selected_profile
    config["project_root"] = str(root_dir)
    return config


def load_scenarios(root_dir: str | Path) -> dict[str, Any]:
    root_dir = Path(root_dir)
    raw = load_yaml(root_dir / "configs" / "scenarios.yaml")
    return raw["scenarios"]
