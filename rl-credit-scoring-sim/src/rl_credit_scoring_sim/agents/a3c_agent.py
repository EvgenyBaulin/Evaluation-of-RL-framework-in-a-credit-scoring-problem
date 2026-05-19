"""
Synchronous Advantage Actor-Critic (A3C-style) agent.

Uses a single shared actor-critic network trained on-policy with full-episode
trajectories. Equivalent to single-worker A3C (= A2C without the SB3 wrapper).
Implemented as a custom BaseController to integrate with the existing pipeline.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_credit_scoring_sim.agents.base import BaseController, TrainingLog
from rl_credit_scoring_sim.utils.device_utils import detect_device
from rl_credit_scoring_sim.utils.progress import TrainingProgress


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)


class A3CController(BaseController):
    """On-policy actor-critic trained with full-episode rollouts."""

    def __init__(self, name: str, config: dict[str, Any], action_dim: int, obs_dim: int) -> None:
        self.name = name
        self.config = config
        self.agent_cfg = config["agents"][name]
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.device = torch.device(config.get("device", detect_device()))
        self.net = _ActorCritic(obs_dim, action_dim, self.agent_cfg["hidden_sizes"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.agent_cfg["learning_rate"])
        self.training_logs: list[dict[str, Any]] = []

    def _rollout(self, env, seed: int, episode: int):
        obs, info = env.reset(seed=seed + episode)
        done = False
        states, actions, rewards, log_probs, values = [], [], [], [], []
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, val = self.net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            states.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(val)
            obs = next_obs
        return states, actions, rewards, log_probs, values, info

    def _update(self, rewards, log_probs, values) -> float:
        gamma = self.agent_cfg["gamma"]
        entropy_coef = self.agent_cfg.get("entropy_coef", 0.01)
        value_coef = self.agent_cfg.get("value_coef", 0.5)

        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(log_probs).squeeze()
        values_t = torch.stack(values).squeeze()
        advantages = returns_t - values_t.detach()

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns_t)

        logits_all = torch.stack([self.net(s)[0] for s in log_probs]).squeeze() if False else None
        entropy_loss = torch.tensor(0.0, device=self.device)

        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def fit(
        self,
        env,
        training_episodes: int,
        seed: int,
        checkpoint_dir: Path,
        checkpoint_frequency: int,
        logging_frequency: int,
    ):
        state_dim = int(self.config.get("state_dim", self.obs_dim))
        with TrainingProgress(
            total_episodes=training_episodes,
            controller=self.name,
            dim=state_dim,
            seed=seed,
        ) as prog:
            for episode in range(training_episodes):
                _, _, rewards, log_probs, values, info = self._rollout(env, seed, episode)
                loss = self._update(rewards, log_probs, values)
                cumulative_reward = float(sum(rewards))
                log = TrainingLog(
                    agent_name=self.name,
                    seed=seed,
                    episode=episode,
                    cumulative_reward=cumulative_reward,
                    scenario_name=info["week_metrics"]["scenario_name"],
                    metadata={"loss": loss},
                )
                self.training_logs.append(asdict(log))
                prog.update(episode=episode, reward=cumulative_reward, loss=loss)
                if (episode + 1) % checkpoint_frequency == 0:
                    self.save(checkpoint_dir / f"{self.name}_seed_{seed}_episode_{episode + 1}.pt")
        return self.training_logs

    def predict(self, observation, deterministic: bool = True):
        obs_t = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(obs_t)
        if deterministic:
            return int(torch.argmax(logits, dim=1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}, path)
