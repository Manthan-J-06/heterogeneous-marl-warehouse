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

    def __init__(self, env: gym.Env, speeds: List[float], capacities: List[int] = None, battery_capacities: List[float] = None) -> None:
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

        # Internal state to track how long each agent has been carrying something
        self.is_carrying: List[bool] = [False] * n_agents
        self.carry_steps: List[int] = [0] * n_agents
        
        # Internal state for battery levels
        self.battery_levels: List[float] = list(self.battery_capacities)

    def reset(self, **kwargs):
        """Reset the environment and internal carry/battery state."""
        self.is_carrying = [False] * len(self.speeds)
        self.carry_steps = [0] * len(self.speeds)
        self.battery_levels = list(self.battery_capacities)
        return self.env.reset(**kwargs)

    def get_battery_levels(self) -> List[float]:
        """Return the current battery levels of each agent."""
        return list(self.battery_levels)

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
        4. Battery depletion: The final effective action costs are deducted:
           - Move actions (0, 1, 2) cost 1.0
           - Load/Unload (3) costs 2.0
           - Idle/No-op (4) costs 0.5

        Returns
        -------
        tuple
            ``(observations, rewards, terminations, truncations, infos)``
            exactly as returned by the wrapped environment.
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

        # 3. Apply battery depletion based on actual executed action
        for agent_id, a in enumerate(effective_actions):
            if self.battery_levels[agent_id] > 0.0:
                if a == 3:
                    cost = 2.0
                elif a == NOOP_ACTION:
                    cost = 0.5
                else:
                    cost = 1.0
                    
                self.battery_levels[agent_id] = max(0.0, self.battery_levels[agent_id] - cost)

        return self.env.step(effective_actions)
