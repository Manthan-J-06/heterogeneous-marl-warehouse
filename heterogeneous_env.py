"""
heterogeneous_env.py
--------------------
A Gymnasium wrapper that simulates heterogeneous agent speeds in a RWARE
multi-agent environment.

Each agent is assigned a speed value in [0.0, 1.0]:
  - speed = 1.0  →  agent always acts (full speed)
  - speed = 0.5  →  50% chance of acting; otherwise a no-op is injected
  - speed = 0.0  →  agent never acts (always no-op)
"""

from __future__ import annotations

import random
from typing import List

import gymnasium as gym

# Action index for "do nothing" in RWARE
NOOP_ACTION = 4


class HeterogeneousRWAREWrapper(gym.Wrapper):
    """Wraps a RWARE Gymnasium environment to simulate heterogeneous agent speeds.

    For each environment step, every agent independently has a
    ``(1 - speed)`` probability of having its chosen action replaced with
    a no-op, effectively simulating a slower agent that skips turns.

    Parameters
    ----------
    env : gym.Env
        The base RWARE Gymnasium environment to wrap.
    speeds : List[float]
        Per-agent speed values, one per agent, each in the range [0.0, 1.0].
        Must match the number of agents in *env*.

    Raises
    ------
    ValueError
        If any speed value is outside [0.0, 1.0], or if the number of speed
        values does not match the number of agents.
    """

    def __init__(self, env: gym.Env, speeds: List[float]) -> None:
        super().__init__(env)

        n_agents = len(env.action_space)
        if len(speeds) != n_agents:
            raise ValueError(
                f"Expected {n_agents} speed values (one per agent), "
                f"but got {len(speeds)}."
            )
        for i, s in enumerate(speeds):
            if not (0.0 <= s <= 1.0):
                raise ValueError(
                    f"Speed for agent {i} must be in [0.0, 1.0], got {s}."
                )

        self.speeds: List[float] = list(speeds)

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def step(self, actions):
        """Apply stochastic no-op injection before forwarding to the env.

        For each agent *i*, with probability ``(1 - speeds[i])`` the
        chosen action is replaced by ``NOOP_ACTION`` (index 4).

        Parameters
        ----------
        actions : list or tuple
            One action per agent, in the same order as ``env.action_space``.

        Returns
        -------
        tuple
            ``(observations, rewards, terminations, truncations, infos)``
            exactly as returned by the wrapped environment — no values are
            modified after the action substitution.
        """
        effective_actions = list(actions)

        for agent_id, speed in enumerate(self.speeds):
            skip_prob = 1.0 - speed  # probability of injecting a no-op
            if random.random() < skip_prob:
                effective_actions[agent_id] = NOOP_ACTION

        return self.env.step(effective_actions)
