from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_credit_scoring_sim.env.simulator import SyntheticCreditSimulator
from rl_credit_scoring_sim.utils.thresholds import build_discrete_action_map


class ThresholdControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        scenarios: dict[str, Any],
        mode: str,
        scale_factor: float,
        seed: int,
        control_mode: str,
        scenario_name: str | None = None,
        scenario_pool: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.scenarios = scenarios
        self.mode = mode
        self.scale_factor = scale_factor
        self.seed_value = seed
        self.control_mode = control_mode
        self.fixed_scenario_name = scenario_name
        self.scenario_pool = scenario_pool or [scenario_name or "base_market"]
        self.rng = np.random.default_rng(seed)
        self.episode_counter = 0
        self.policy_config = config["policy"]
        self.action_map = build_discrete_action_map(self.policy_config)
        self.simulator = SyntheticCreditSimulator(config, scenarios, scale_factor=scale_factor, seed=seed)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.simulator.max_observation_dim,),
            dtype=np.float32,
        )
        if control_mode == "discrete":
            self.action_space = spaces.Discrete(len(self.action_map))
        else:
            action_dim = 2 if self.policy_config["split_policy"] else 1
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def _select_scenario_name(self) -> str:
        if self.fixed_scenario_name is not None:
            return self.fixed_scenario_name
        if self.config["environment"]["scenario_randomization"] and self.mode == "train":
            return self.scenario_pool[self.episode_counter % len(self.scenario_pool)]
        return self.scenario_pool[0]

    def _continuous_to_thresholds(self, action: np.ndarray) -> tuple[float, float]:
        action = np.asarray(action, dtype=float).reshape(-1)
        threshold_min = self.policy_config["threshold_min"]
        threshold_max = self.policy_config["threshold_max"]
        if not self.policy_config["split_policy"]:
            action = np.array([action[0], action[0]], dtype=float)
        if action.size == 1:
            action = np.array([action[0], action[0]], dtype=float)
        action = np.clip(action, -1.0, 1.0)
        scaled = threshold_min + ((action + 1.0) / 2.0) * (threshold_max - threshold_min)
        granularity = self.policy_config["threshold_granularity"]
        scaled = np.round(scaled / granularity) * granularity
        scaled = np.clip(scaled, threshold_min, threshold_max)
        return float(scaled[0]), float(scaled[1])

    def action_to_thresholds(self, action: Any) -> tuple[float, float]:
        if self.control_mode == "discrete":
            return self.action_map[int(action)]
        return self._continuous_to_thresholds(np.asarray(action, dtype=float))

    def thresholds_to_action(self, thresholds: tuple[float, float]) -> Any:
        if self.control_mode == "discrete":
            return self.action_map.index((float(thresholds[0]), float(thresholds[1])))
        threshold_min = self.policy_config["threshold_min"]
        threshold_max = self.policy_config["threshold_max"]
        values = np.asarray(thresholds, dtype=float)
        scaled = 2.0 * (values - threshold_min) / max(1.0, threshold_max - threshold_min) - 1.0
        if not self.policy_config["split_policy"]:
            return scaled[:1].astype(np.float32)
        return scaled.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)
        options = options or {}
        scenario_name = options.get("scenario_name") or self._select_scenario_name()
        episode_seed = int(self.seed_value + 997 * self.episode_counter)
        self.simulator.reset_episode(scenario_name=scenario_name, seed=episode_seed)
        self.episode_counter += 1
        return self.simulator.get_observation(), self.simulator.get_reset_info()

    def step(self, action):
        thresholds = self.action_to_thresholds(action)
        observation, reward, done, info = self.simulator.step(thresholds)
        info["thresholds"] = thresholds
        return observation, reward, done, False, info

    def preview_action(self, thresholds: tuple[float, float]) -> dict[str, float]:
        return self.simulator.score_action_candidates(thresholds)

    def get_week_metrics(self) -> dict[str, Any]:
        return self.simulator.last_week_metrics
