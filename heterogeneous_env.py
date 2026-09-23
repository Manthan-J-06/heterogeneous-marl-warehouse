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

import numpy as np
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
    capacities : List[int], optional
        Per-agent capacity values (max consecutive loaded steps).
        Defaults to effectively infinite for backward compatibility.
    battery_capacities : List[float], optional
        Per-agent battery capacity values (starting energy budget).
        Defaults to effectively infinite for backward compatibility.

    Raises
    ------
    ValueError
        If any speed value is outside [0.0, 1.0], or if the number of speed,
        capacity, or battery values does not match the number of agents.
    """

    def __init__(self, env: gym.Env, speeds: List[float], capacities: List[int] = None, battery_capacities: List[float] = None, energy_weight: float = 0.0) -> None:
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

        if capacities is None:
            capacities = [int(1e9)] * n_agents
        elif len(capacities) != n_agents:
            raise ValueError(
                f"Expected {n_agents} capacity values, "
                f"but got {len(capacities)}."
            )

        if battery_capacities is None:
            battery_capacities = [1e9] * n_agents
        elif len(battery_capacities) != n_agents:
            raise ValueError(
                f"Expected {n_agents} battery capacity values, "
                f"but got {len(battery_capacities)}."
            )

        self.speeds: List[float] = list(speeds)
        self.capacities: List[int] = list(capacities)
        self.battery_capacities: List[float] = list(battery_capacities)
        self.energy_weight: float = float(energy_weight)

        # --- Convenience properties expected by training loops ---
        self.n_agents: int = n_agents
        self.n_actions: int = self.env.action_space[0].n
        obs_space_0 = self.env.observation_space[0]
        self.obs_dim: int = obs_space_0.shape[0] if hasattr(obs_space_0, 'shape') else obs_space_0.n

        # Internal state to track how long each agent has been carrying something
        self.is_carrying: List[bool] = [False] * n_agents
        self.carry_steps: List[int] = [0] * n_agents
        
        # Internal state for battery levels and energy metrics
        self.battery_levels: List[float] = list(self.battery_capacities)
        self.energy_consumed: List[float] = [0.0] * n_agents

    def reset(self, **kwargs):
        """Reset the environment and internal carry/battery state."""
        self.is_carrying = [False] * len(self.speeds)
        self.carry_steps = [0] * len(self.speeds)
        self.battery_levels = list(self.battery_capacities)
        self.energy_consumed = [0.0] * len(self.speeds)
        return self.env.reset(**kwargs)

    def get_battery_levels(self) -> List[float]:
        """Return the current battery levels of each agent."""
        return list(self.battery_levels)

    def get_energy_consumed(self) -> List[float]:
        """Return the total energy consumed by each agent since reset."""
        return list(self.energy_consumed)

    def global_state(self, obs_array) -> np.ndarray:
        """Flat concatenation of all per-agent observations.

        Used by QMIX's mixing network and MAPPO's centralized critic.
        Identical contract to RwareEnvWrapper.global_state.
        """
        return np.asarray(obs_array, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def step(self, actions):
        """Apply stochastic no-op injection, capacity blocks, and battery depletion.

        1. Dead state: If battery <= 0, action is forced to NOOP_ACTION.
        2. Speeds: For each agent *i*, with probability ``(1 - speeds[i])``
           the action is replaced by ``NOOP_ACTION``.
        3. Capacities: If an agent's ``carry_steps`` exceed ``capacities[i]``,
           any remaining action other than Unload (3) is replaced by
           ``NOOP_ACTION`` to force an unload.
        4. Battery depletion & Energy Penalty: The final effective action costs:
           - Move actions (0, 1, 2) cost 1.0
           - Load/Unload (3) costs 2.0
           - Idle/No-op (4) costs 0.5
           The cost is deducted from battery_levels and `energy_weight * cost`
           is subtracted from each agent's reward.

        Returns
        -------
        tuple
            ``(observations, rewards, terminations, truncations, infos)``
            exactly as returned by the wrapped environment, with rewards adjusted.
        """
        # Expose block counts for validation/metrics this step
        self.capacity_blocks_this_step = [0] * len(self.speeds)
        self.dead_blocks_this_step = [0] * len(self.speeds)
        
        effective_actions = list(actions)

        # 0. Dead check logic (new)
        for agent_id, battery in enumerate(self.battery_levels):
            if battery <= 0.0:
                effective_actions[agent_id] = NOOP_ACTION
                self.dead_blocks_this_step[agent_id] = 1

        # 1. Speed check logic (existing)
        for agent_id, speed in enumerate(self.speeds):
            # purely visual note: if they were already dead, this NOOP doesn't matter,
            # but we still roll for it.
            skip_prob = 1.0 - speed
            if random.random() < skip_prob:
                effective_actions[agent_id] = NOOP_ACTION

        # 2. Capacity check logic (existing)
        for agent_id, cp in enumerate(self.capacities):
            a = effective_actions[agent_id]
            if self.is_carrying[agent_id]:
                # Update carry steps since they started this step carrying
                self.carry_steps[agent_id] += 1
                
                if a == 3:
                    # Taking action 3 while carrying = dropping the load
                    self.is_carrying[agent_id] = False
                    self.carry_steps[agent_id] = 0
                else:
                    # If they are over capacity and NOT dropping it, block action!
                    if self.carry_steps[agent_id] > cp:
                        if a != NOOP_ACTION:
                            effective_actions[agent_id] = NOOP_ACTION
                            self.capacity_blocks_this_step[agent_id] = 1
            else:
                if a == 3:
                    # Taking action 3 while empty = picking up load (if over a shelf)
                    # We assume action 3 unconditionally toggles the 'carrying' state
                    # for simulation/capacity-tracking purposes.
                    self.is_carrying[agent_id] = True
                    self.carry_steps[agent_id] = 0

        # Execute step in base env
        obs, rewards, terminations, truncations, infos = self.env.step(effective_actions)
        
        adjusted_rewards = list(rewards)

        # 3. Apply battery depletion and reward penalty based on actual executed action
        for agent_id, a in enumerate(effective_actions):
            if self.battery_levels[agent_id] > 0.0:
                if a == 3:
                    cost = 2.0
                elif a == NOOP_ACTION:
                    cost = 0.5
                else:
                    cost = 1.0
                    
                self.battery_levels[agent_id] = max(0.0, self.battery_levels[agent_id] - cost)
                self.energy_consumed[agent_id] += cost
                
                # Apply penalty to reward
                adjusted_rewards[agent_id] -= (self.energy_weight * cost)

        return (obs, tuple(adjusted_rewards), terminations, truncations, infos)

def build_heterogeneous_env(config: dict) -> gym.Env:
    """Builds and wraps the environment based on the configuration dict.

    Parameters
    ----------
    config : dict
        A configuration dictionary matching the structure of `configs/*.yaml`.
        Must contain 'environment' and 'heterogeneity' sections.

    Returns
    -------
    gym.Env
        The wrapped heterogeneous Gymnasium environment.
    """
    env_cfg = config.get("environment", {})
    env_id = env_cfg.get("env_id", "rware-tiny-2ag-v2")

    base_env = gym.make(env_id)
    n_agents = len(base_env.action_space)

    het_cfg = config.get("heterogeneity", {})
    enabled = het_cfg.get("enabled", False)
    agents_cfg = het_cfg.get("agents", [])

    energy_weight = het_cfg.get("energy_weight", 0.0)

    if not enabled or len(agents_cfg) != n_agents:
        # Fallback to homogeneous / unconstrained defaults
        speeds = [1.0] * n_agents
        capacities = [999999] * n_agents
        battery_capacities = [999999.0] * n_agents
    else:
        speeds = [float(a.get("speed", 1.0)) for a in agents_cfg]
        capacities = [int(a.get("capacity", 999999)) for a in agents_cfg]
        battery_capacities = [float(a.get("battery_capacity", 999999.0)) for a in agents_cfg]

    return HeterogeneousRWAREWrapper(
        base_env,
        speeds=speeds,
        capacities=capacities,
        battery_capacities=battery_capacities,
        energy_weight=energy_weight
    )

