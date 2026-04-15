from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_credit_scoring_sim.agents.base import BaseController, TrainingLog


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage = deque(maxlen=capacity)

    def add(self, transition: tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.storage.append(transition)

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.choice(len(self.storage), size=batch_size, replace=False)
        batch = [self.storage[idx] for idx in indices]
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        )


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CustomDQNController(BaseController):
    def __init__(self, name: str, config: dict[str, Any], action_dim: int, obs_dim: int, double: bool) -> None:
        self.name = name
        self.config = config
        self.agent_cfg = config["agents"][name]
        self.double = double
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_net = QNetwork(obs_dim, action_dim, self.agent_cfg["hidden_sizes"]).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, self.agent_cfg["hidden_sizes"]).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.agent_cfg["learning_rate"])
        self.buffer = ReplayBuffer(self.agent_cfg["replay_size"])
        self.training_logs: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(0)
        self.total_steps = 0

    def _epsilon(self, max_steps: int) -> float:
        start = self.agent_cfg["epsilon_start"]
        end = self.agent_cfg["epsilon_end"]
        decay_steps = max(1, int(max_steps * self.agent_cfg["epsilon_decay_fraction"]))
        if self.total_steps >= decay_steps:
            return end
        fraction = self.total_steps / decay_steps
        return start + fraction * (end - start)

    def _select_action(self, observation: np.ndarray, deterministic: bool, max_steps: int) -> int:
        epsilon = 0.0 if deterministic else self._epsilon(max_steps)
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.action_dim))
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(obs_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def _train_step(self, batch_size: int, gamma: float) -> None:
        obs, action, reward, next_obs, done = self.buffer.sample(batch_size, self._rng)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(obs_tensor).gather(1, action_tensor)
        with torch.no_grad():
            if self.double:
                next_action = self.online_net(next_obs_tensor).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs_tensor).gather(1, next_action)
            else:
                next_q = self.target_net(next_obs_tensor).max(dim=1, keepdim=True).values
            target = reward_tensor + gamma * (1.0 - done_tensor) * next_q
        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

    def fit(
        self,
        env,
        training_episodes: int,
        seed: int,
        checkpoint_dir: Path,
        checkpoint_frequency: int,
        logging_frequency: int,
    ):
        self._rng = np.random.default_rng(seed)
        max_steps = training_episodes * self.config["environment"]["horizon_weeks"]
        learning_starts = min(self.agent_cfg["learning_starts"], max(50, max_steps // 4))
        gamma = self.agent_cfg["gamma"]
        batch_size = self.agent_cfg["batch_size"]
        train_frequency = self.agent_cfg["train_frequency"]
        gradient_steps = self.agent_cfg["gradient_steps"]
        target_update_interval = self.agent_cfg["target_update_interval"]

        for episode in range(training_episodes):
            obs, info = env.reset(seed=seed + episode)
            done = False
            cumulative_reward = 0.0
            while not done:
                action = self._select_action(obs, deterministic=False, max_steps=max_steps)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.buffer.add((obs, action, reward, next_obs, done))
                obs = next_obs
                cumulative_reward += reward
                self.total_steps += 1

                if len(self.buffer) >= batch_size and self.total_steps >= learning_starts and self.total_steps % train_frequency == 0:
                    for _ in range(gradient_steps):
                        self._train_step(batch_size=batch_size, gamma=gamma)

                if self.total_steps % target_update_interval == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

            log = TrainingLog(
                agent_name=self.name,
                seed=seed,
                episode=episode,
                cumulative_reward=float(cumulative_reward),
                scenario_name=info["week_metrics"]["scenario_name"],
                metadata={"total_steps": self.total_steps},
            )
            self.training_logs.append(asdict(log))
            if (episode + 1) % checkpoint_frequency == 0:
                self.save(checkpoint_dir / f"{self.name}_seed_{seed}_episode_{episode + 1}.pt")
            if (episode + 1) % logging_frequency == 0:
                pass
        self.target_net.load_state_dict(self.online_net.state_dict())
        return self.training_logs

    def predict(self, observation, deterministic: bool = True):
        return self._select_action(np.asarray(observation, dtype=np.float32), deterministic=deterministic, max_steps=max(1, self.total_steps))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_logs": self.training_logs,
                "total_steps": self.total_steps,
                "config": self.agent_cfg,
                "double": self.double,
            },
            path,
        )
