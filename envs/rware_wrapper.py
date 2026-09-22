"""
Thin wrapper around RWARE (Multi-Robot Warehouse) confirming a consistent,
verified interface for both QMIX and MAPPO training loops.

Verified against rware==2.0.0 + gymnasium directly in a sandbox before
shipping this file — the raw env has two gotchas this wrapper fixes:
  1. env.step() requires a list of plain Python ints, NOT a tuple of
     np.int64 (passing np.int64 raises `ValueError: ... is not a valid
     Action` inside rware's internal Action enum lookup).
  2. Reward comes back as a list of np.float64 per agent, not a single
     scalar — fine for MARL, but worth knowing before you assume
     Gymnasium's single-agent step signature applies unmodified.
"""

import numpy as np
import gymnasium as gym
import rware  # noqa: F401  (registers rware-* env ids on import)


class RwareEnvWrapper:
    """
    Homogeneous-fleet RWARE wrapper.

    env_id examples: 'rware-tiny-2ag-easy-v2', 'rware-tiny-4ag-easy-v2'
    (N in 'Nag' must match n_agents you configure elsewhere).
    """

    def __init__(self, env_id: str, seed: int = 0):
        self.env = gym.make(env_id)
        self.n_agents = self.env.action_space.n if hasattr(self.env.action_space, "n") \
            else len(self.env.action_space.spaces)
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.n_actions = self.env.action_space[0].n
        self._seed = seed

    def reset(self):
        obs, info = self.env.reset(seed=self._seed)
        return self._to_array(obs), info

    def step(self, actions):
        """actions: iterable of per-agent ints (numpy ints are fine, cast happens here)."""
        actions = [int(a) for a in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._to_array(obs)
        reward = np.asarray(reward, dtype=np.float32)
        # RWARE returns a single bool for terminated/truncated (shared episode end)
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info

    def global_state(self, obs_array: np.ndarray) -> np.ndarray:
        """Concatenation of all per-agent observations — used by the QMIX mixing
        network and the MAPPO centralized critic. Swap this out for a smaller
        true global state later if the concatenated size becomes a bottleneck."""
        return obs_array.reshape(-1)

    @staticmethod
    def _to_array(obs_tuple) -> np.ndarray:
        return np.stack([np.asarray(o, dtype=np.float32) for o in obs_tuple], axis=0)

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # Smoke test — run directly with `python envs/rware_wrapper.py`
    env = RwareEnvWrapper("rware-tiny-4ag-easy-v2", seed=0)
    obs, _ = env.reset()
    print("obs shape (n_agents, obs_dim):", obs.shape)
    print("n_agents:", env.n_agents, "n_actions:", env.n_actions, "obs_dim:", env.obs_dim)
    for t in range(5):
        actions = [np.random.randint(env.n_actions) for _ in range(env.n_agents)]
        obs, reward, done, info = env.step(actions)
        print(f"  step {t}: reward={reward}, done={done}")
    print("Global state shape:", env.global_state(obs).shape)
    env.close()
    print("RWARE wrapper smoke test passed.")
