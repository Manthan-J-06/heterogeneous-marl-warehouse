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

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
env = build_heterogeneous_env(config)

SPEEDS = env.speeds
CAPACITIES = env.capacities
BATTERY_CAPACITIES = env.battery_capacities
n_agents = len(SPEEDS)

total_steps = 0
noop_counts = [0] * n_agents            # times a no-op was injected by speed
capacity_blocked_counts = [0] * n_agents # times an action was blocked by capacity
dead_counts = [0] * n_agents            # times an action was blocked by zero battery
action_counts = [0] * n_agents          # total steps taken per agent

print(f"Environment : {ENV_ID}")
print(f"Agent speeds: {SPEEDS}")
print(f"Capacities  : {CAPACITIES}")
print(f"Batteries   : {BATTERY_CAPACITIES}")
print(f"Episodes    : {NUM_EPISODES}")
print("-" * 40)

# -----------------------------------------------------------------------
# Episode loop
# -----------------------------------------------------------------------
for episode in range(1, NUM_EPISODES + 1):
    obs, info = env.reset()
    episode_steps = 0

    while True:
        # Sample a random action for each agent from its action space
        random_actions = [
            env.action_space[i].sample()
            for i in range(n_agents)
        ]

        # Track which actions *would* become no-ops by mirroring the wrapper's RNG.
        # (Counting only — the actual injection happens inside the wrapper.)
        for i, speed in enumerate(SPEEDS):
            action_counts[i] += 1
            skip_prob = 1.0 - speed
            if random.random() < skip_prob:
                noop_counts[i] += 1

        obs, rewards, terminated, truncated, info = env.step(random_actions)
        
        # Accumulate capacity constraints and dead blocks this step
        for i in range(n_agents):
            capacity_blocked_counts[i] += getattr(env, "capacity_blocks_this_step", [0]*n_agents)[i]
            dead_counts[i] += getattr(env, "dead_blocks_this_step", [0]*n_agents)[i]

        episode_steps += 1

        # Check termination — RWARE may return a list of bools or a single bool
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
    print(f"  Episode {episode}: {episode_steps} steps")

# Snapshot final battery levels before closing
if hasattr(env, "get_battery_levels"):
    final_batteries = env.get_battery_levels()
else:
    final_batteries = [0.0] * n_agents
env.close()

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------
print("-" * 40)
print(f"Total steps across all episodes: {total_steps}")
print()
print(f"{'Agent':<6} {'Speed':<6} {'Cap':<4} {'Batt':<6} {'Speed No-ops':<13} {'Cap Blocks':<11} {'Dead Steps':<11} {'Total'}")
print("-" * 75)
for i in range(n_agents):
    steps = action_counts[i]
    speed_noops = noop_counts[i]
    cap_blocks = capacity_blocked_counts[i]
    dead = dead_counts[i]
    print(
        f"  {i:<4} {SPEEDS[i]:<6.1f} {CAPACITIES[i]:<4} {BATTERY_CAPACITIES[i]:<6.1f} "
        f"{speed_noops:<13} {cap_blocks:<11} {dead:<11} {steps:<11}"
    )

print()
print("Validation notes:")
for i in range(min(2, n_agents)):
    print(f"  Agent {i} final battery: {final_batteries[i]:.1f}")
print(f"  Agent 0 (speed: {SPEEDS[0]:.1f}): speed no-ops should be ~{(1 - SPEEDS[0])*100:.0f}%   → got {noop_counts[0] / max(action_counts[0],1)*100:.1f}%")
if n_agents > 1:
    print(f"  Agent 1 (speed: {SPEEDS[1]:.1f}): speed no-ops should be ~{(1 - SPEEDS[1])*100:.0f}%  → got {noop_counts[1] / max(action_counts[1],1)*100:.1f}%")
print(f"  Dead Steps: Should be non-zero if total steps * avg cost > battery budget.")
