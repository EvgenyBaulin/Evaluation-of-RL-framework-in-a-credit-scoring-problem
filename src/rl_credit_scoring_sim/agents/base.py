from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingLog:
    agent_name: str
    seed: int
    episode: int
    cumulative_reward: float
    scenario_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseController:
    name: str

    def fit(self, env, training_episodes: int, seed: int, checkpoint_dir: Path, checkpoint_frequency: int, logging_frequency: int):
        raise NotImplementedError

    def predict(self, observation, deterministic: bool = True):
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError
