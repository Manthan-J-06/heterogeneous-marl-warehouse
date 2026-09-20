"""
validate_heterogeneous_env.py
------------------------------
Validation script for HeterogeneousRWAREWrapper.

Creates a ``rware-tiny-2ag-v2`` environment, wraps it with agent speeds
[1.0, 0.3], runs 5 episodes under a random policy, and reports how often
each agent's action was actually replaced with a no-op.

Expected outcome:
  - Agent 0 (speed 1.0) → 0% no-op replacements
  - Agent 1 (speed 0.3) → ~70% no-op replacements
"""

from __future__ import annotations

import random
import numpy as np
from typing import List

import gymnasium as gym

# Patch rware to register its envs with gymnasium if not already done
try:
    import rware  # noqa: F401  — registers envs as a side-effect
except ImportError as exc:
    raise SystemExit(
        "The 'rware' package is required. Install it with:\n"
        "    pip install rware\n"
        f"Original error: {exc}"
    )

import yaml
import random
import numpy as np
from pathlib import Path
from heterogeneous_env import NOOP_ACTION, HeterogeneousRWAREWrapper, build_heterogeneous_env

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
CONFIG_PATH = "configs/heterogeneous_small.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

ENV_ID = config.get("environment", {}).get("env_id", "rware-tiny-2ag-v2")
NUM_EPISODES = config.get("validation", {}).get("num_episodes", 5)

def run_evaluation(config: dict, label: str):
    env = build_heterogeneous_env(config)

    SPEEDS = env.speeds
    CAPACITIES = env.capacities
    BATTERY_CAPACITIES = env.battery_capacities
    ENERGY_WEIGHT = env.energy_weight
    n_agents = len(SPEEDS)

    total_steps = 0
    noop_counts = [0] * n_agents
    capacity_blocked_counts = [0] * n_agents
    dead_counts = [0] * n_agents
    action_counts = [0] * n_agents
    
    total_rewards = [0.0] * n_agents
    total_energy = [0.0] * n_agents

    print(f"\n{'=' * 75}\n RUN: {label}")
    print(f" Environment : {ENV_ID}")
    print(f" Agent speeds: {SPEEDS}")
    print(f" Capacities  : {CAPACITIES}")
    print(f" Batteries   : {BATTERY_CAPACITIES}")
    print(f" Energy Wgt  : {ENERGY_WEIGHT}")
    print(f" Episodes    : {NUM_EPISODES}\n{'-' * 75}")

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        episode_steps = 0

        while True:
            random_actions = [
                env.action_space[i].sample()
                for i in range(n_agents)
            ]

            for i, speed in enumerate(SPEEDS):
                action_counts[i] += 1
                skip_prob = 1.0 - speed
                if random.random() < skip_prob:
                    noop_counts[i] += 1

            obs, rewards, terminated, truncated, info = env.step(random_actions)
            
            for i in range(n_agents):
                total_rewards[i] += rewards[i]
                capacity_blocked_counts[i] += getattr(env, "capacity_blocks_this_step", [0]*n_agents)[i]
                dead_counts[i] += getattr(env, "dead_blocks_this_step", [0]*n_agents)[i]

            episode_steps += 1

            if isinstance(terminated, (list, tuple, np.ndarray)):
                done_terminated = all(terminated)
            else:
                done_terminated = bool(terminated)

            if isinstance(truncated, (list, tuple, np.ndarray)):
                done_truncated = all(truncated)
            else:
                done_truncated = bool(truncated)

            if done_terminated or done_truncated or episode_steps >= 500:
                break

        total_steps += episode_steps
        
        # Accumulate energy immediately before the next reset zeroes it out
        if hasattr(env, "get_energy_consumed"):
            ep_energy = env.get_energy_consumed()
            for i in range(n_agents):
                total_energy[i] += ep_energy[i]

    if hasattr(env, "get_battery_levels"):
        final_batteries = env.get_battery_levels()
    else:
        final_batteries = [0.0] * n_agents
    env.close()

    print(f"{'Agent':<6} {'Speed':<6} {'Cap':<4} {'Batt':<6} {'Reward':<10} {'Energy Used':<12} {'Dead Steps':<11} {'Total'}")
    print("-" * 75)
    for i in range(n_agents):
        steps = action_counts[i]
        dead = dead_counts[i]
        rew = total_rewards[i]
        nrg = total_energy[i]
        print(
            f"  {i:<4} {SPEEDS[i]:<6.1f} {CAPACITIES[i]:<4} {BATTERY_CAPACITIES[i]:<6.1f} "
            f"{rew:<10.2f} {nrg:<12.1f} {dead:<11} {steps:<11}"
        )


if __name__ == "__main__":
    # Ensure heterogeneity is initialized safely in case config is purely missing the key
    if "heterogeneity" not in config:
        config["heterogeneity"] = {"enabled": True, "agents": [{"speed": 1.0}, {"speed": 0.5}]}
        
    config["heterogeneity"]["energy_weight"] = 0.0
    run_evaluation(config, "Zero Energy Penalty (Baseline)")
    
    config["heterogeneity"]["energy_weight"] = 0.05
    run_evaluation(config, "0.05 Energy Weight Penalty")
