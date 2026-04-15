from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_credit_scoring_sim.agents.base import BaseController


class SB3Controller(BaseController):
    def __init__(self, name: str, config: dict[str, Any], algorithm_cls) -> None:
        self.name = name
        self.config = config
        self.agent_cfg = config["agents"][name]
        self.algorithm_cls = algorithm_cls
        self.model = None

    def fit(
        self,
        env,
        training_episodes: int,
        seed: int,
        checkpoint_dir: Path,
        checkpoint_frequency: int,
        logging_frequency: int,
    ):
        total_timesteps = training_episodes * self.config["environment"]["horizon_weeks"]
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, checkpoint_frequency * self.config["environment"]["horizon_weeks"]),
            save_path=str(checkpoint_dir),
            name_prefix=f"{self.name}_seed_{seed}",
        )
        if self.algorithm_cls is PPO:
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                learning_rate=self.agent_cfg["learning_rate"],
                n_steps=min(self.agent_cfg["n_steps"], total_timesteps),
                batch_size=min(self.agent_cfg["batch_size"], max(8, total_timesteps)),
                gamma=self.agent_cfg["gamma"],
                gae_lambda=self.agent_cfg["gae_lambda"],
                ent_coef=self.agent_cfg["ent_coef"],
                clip_range=self.agent_cfg["clip_range"],
                policy_kwargs=self.agent_cfg["policy_kwargs"],
            )
        elif self.algorithm_cls is SAC:
            learning_starts = min(self.agent_cfg["learning_starts"], max(50, total_timesteps // 4))
            self.model = SAC(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                learning_rate=self.agent_cfg["learning_rate"],
                buffer_size=self.agent_cfg["buffer_size"],
                learning_starts=learning_starts,
                batch_size=min(self.agent_cfg["batch_size"], max(16, total_timesteps)),
                gamma=self.agent_cfg["gamma"],
                tau=self.agent_cfg["tau"],
                train_freq=self.agent_cfg["train_freq"],
                gradient_steps=self.agent_cfg["gradient_steps"],
                policy_kwargs=self.agent_cfg["policy_kwargs"],
            )
        else:
            raise ValueError(f"Unsupported SB3 algorithm for controller {self.name}")
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=False)
        return []

    def predict(self, observation, deterministic: bool = True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
