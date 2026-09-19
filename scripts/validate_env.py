"""
validate_env.py — Phase 1 Random-Policy Environment Validation

Loads the RWARE warehouse environment specified in configs/env_small.yaml,
runs N episodes with a random policy (uniform action-space sampling), and
confirms that reset(), step(), and termination all work correctly.

Usage:
    python scripts/validate_env.py
    python scripts/validate_env.py --config configs/env_small.yaml

Exit code 0 on success, non-zero on any error.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
import numpy as np

import gymnasium as gym
import rware  # noqa: F401  — registers RWARE envs with gymnasium

import sys
from pathlib import Path
# Add src to the Python path so we can import our new module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.metrics_logger import MetricsLogger


def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_env_info(env: gym.Env, env_id: str) -> None:
    """Print key details about the environment for documentation."""
    print("=" * 70)
    print(f"  Environment: {env_id}")
    print(f"  Number of agents: {env.unwrapped.n_agents}")
    print(f"  Observation space (per agent): {env.observation_space[0]}")
    print(f"  Action space (per agent): {env.action_space[0]}")
    print(f"  Max steps (from env): {getattr(env.unwrapped, 'max_steps', 'N/A')}")
    print("=" * 70)
    print()


def run_validation(config: dict) -> list[dict]:
    """
    Run random-policy episodes and return per-episode statistics.

    Each episode dict contains:
        - episode: int (1-indexed)
        - total_reward: float (sum of per-agent rewards)
        - steps: int
        - terminated: bool (True if any agent terminated naturally)
        - truncated: bool (True if episode was truncated by step limit)
    """
    env_cfg = config["environment"]
    val_cfg = config["validation"]

    env_id = env_cfg["env_id"]
    num_episodes = val_cfg["num_episodes"]
    seed = val_cfg["seed"]

    # Create the environment
    print(f"Creating environment: {env_id} ...")
    env = gym.make(env_id)
    print_env_info(env, env_id)

    n_agents = env.unwrapped.n_agents
    results = []
    
    # Initialize our shiny new logger!
    logger = MetricsLogger(n_agents)

    for ep in range(1, num_episodes + 1):
        # Reset with a deterministic seed per episode for reproducibility
        obs, info = env.reset(seed=seed + ep)

        # Validate observation structure
        assert isinstance(obs, tuple), (
            f"Expected obs to be a tuple of per-agent observations, got {type(obs)}"
        )
        assert len(obs) == n_agents, (
            f"Expected {n_agents} observations, got {len(obs)}"
        )

        # Reset our logger for the new episode
        logger.reset_episode()
        
        ep_steps = 0
        ep_terminated = False
        ep_truncated = False

        while True:
            # Random policy: sample one action per agent
            actions = tuple(
                env.action_space[i].sample() for i in range(n_agents)
            )

            obs, rewards, terminated, truncated, info = env.step(actions)

            # Validate step outputs
            assert len(rewards) == n_agents, (
                f"Step {ep_steps}: expected {n_agents} rewards, got {len(rewards)}"
            )

            # Let the logger do the math!
            logger.log_step(rewards)
            
            ep_steps += 1

            # Check termination: RWARE returns per-agent done flags
            if isinstance(terminated, (list, tuple, np.ndarray)):
                done_terminated = all(terminated)
            else:
                done_terminated = bool(terminated)

            if isinstance(truncated, (list, tuple, np.ndarray)):
                done_truncated = all(truncated)
            else:
                done_truncated = bool(truncated)

            if done_terminated or done_truncated:
                ep_terminated = done_terminated
                ep_truncated = done_truncated
                break

        # Grab the total reward from the logger so we can still print it
        summary = logger.get_episode_summary()
        
        results.append({
            "episode": ep,
            "total_reward": summary["team_reward"],
            "steps": ep_steps,
            "terminated": ep_terminated,
            "truncated": ep_truncated,
        })

        # Print the detailed breakdown for this episode!
        logger.print_summary(ep)

    env.close()
    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary table of all episodes."""
    print()
    print("=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    header = (
        f"{'Episode':<10} | {'Steps':<8} | {'Total Reward':<14} | {'Outcome':<12}"
    )
    print(header)
    print("-" * len(header))

    total_steps = 0
    total_reward = 0.0
    for r in results:
        outcome = "TERMINATED" if r["terminated"] else "TRUNCATED"
        print(
            f"{r['episode']:<10} | {r['steps']:<8} | "
            f"{r['total_reward']:<14.2f} | {outcome:<12}"
        )
        total_steps += r["steps"]
        total_reward += r["total_reward"]

    print("-" * len(header))
    n = len(results)
    print(f"{'TOTAL':<10} | {total_steps:<8} | {total_reward:<14.2f} |")
    print(f"{'MEAN':<10} | {total_steps / n:<8.1f} | {total_reward / n:<14.2f} |")
    print()
    print(f"  All {n} episodes completed successfully.")
    print(f"  Environment resets, steps, and terminates correctly.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Validate RWARE environment with random-policy rollouts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/env_small.yaml",
        help="Path to the environment configuration YAML file.",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  PHASE 1 - RWARE ENVIRONMENT VALIDATION")
    print("  Random-policy rollout test")
    print("=" * 70)
    print()

    config = load_config(args.config)

    start_time = time.time()
    results = run_validation(config)
    elapsed = time.time() - start_time

    print_summary(results)
    print(f"  Total wall time: {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    main()
