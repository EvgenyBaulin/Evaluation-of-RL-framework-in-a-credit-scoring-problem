from __future__ import annotations

from stable_baselines3 import A2C, PPO, SAC

from rl_credit_scoring_sim.agents.a3c_agent import A3CController
from rl_credit_scoring_sim.agents.custom_dqn import CustomDQNController
from rl_credit_scoring_sim.agents.sb3_wrappers import SB3Controller


def controller_mode(name: str) -> str:
    return "discrete" if name in {"dqn", "double_dqn", "a3c"} else "continuous"


def make_agent(name: str, config: dict, action_dim: int, obs_dim: int):
    if name == "dqn":
        return CustomDQNController(name=name, config=config, action_dim=action_dim, obs_dim=obs_dim, double=False)
    if name == "double_dqn":
        return CustomDQNController(name=name, config=config, action_dim=action_dim, obs_dim=obs_dim, double=True)
    if name == "a3c":
        return A3CController(name=name, config=config, action_dim=action_dim, obs_dim=obs_dim)
    if name == "a2c":
        return SB3Controller(name=name, config=config, algorithm_cls=A2C)
    if name == "ppo":
        return SB3Controller(name=name, config=config, algorithm_cls=PPO)
    if name == "sac":
        return SB3Controller(name=name, config=config, algorithm_cls=SAC)
    raise KeyError(f"Unknown RL agent: {name}")
