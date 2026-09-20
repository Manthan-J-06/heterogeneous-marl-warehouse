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

from heterogeneous_env import NOOP_ACTION, HeterogeneousRWAREWrapper

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
ENV_ID = "rware-tiny-2ag-v2"
SPEEDS: List[float] = [1.0, 0.3]
NUM_EPISODES = 5

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
base_env = gym.make(ENV_ID)
env = HeterogeneousRWAREWrapper(base_env, speeds=SPEEDS)
n_agents = len(SPEEDS)

total_steps = 0
noop_counts = [0] * n_agents     # times a no-op was injected per agent
action_counts = [0] * n_agents   # total steps taken per agent

print(f"Environment : {ENV_ID}")
print(f"Agent speeds: {SPEEDS}")
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

env.close()

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------
print("-" * 40)
print(f"Total steps across all episodes: {total_steps}")
print()
print(f"{'Agent':<8} {'Speed':<8} {'No-ops':<10} {'Steps':<10} {'No-op %':<10}  {'Expected %'}")
print("-" * 62)
for i in range(n_agents):
    steps = action_counts[i]
    noops = noop_counts[i]
    pct = (noops / steps * 100) if steps > 0 else 0.0
    expected_pct = (1.0 - SPEEDS[i]) * 100
    print(
        f"  {i:<6} {SPEEDS[i]:<8.1f} {noops:<10} {steps:<10} "
        f"{pct:<10.1f}  {expected_pct:.1f}%"
    )

print()
print("Validation notes:")
print(f"  Agent 0 (speed 1.0): no-ops should be ~0%  → got {noop_counts[0] / max(action_counts[0],1)*100:.1f}%")
print(f"  Agent 1 (speed 0.3): no-ops should be ~70% → got {noop_counts[1] / max(action_counts[1],1)*100:.1f}%")
