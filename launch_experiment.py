"""
launch_experiment.py
---------------------
Config-driven experiment launcher.

Given a config file, this script:
1. Loads the config
2. Builds the per-agent fleet (uniform or explicit mode)
3. Converts the fleet into the format the environment wrapper expects
4. Builds the heterogeneous environment
5. Runs a short test episode to confirm everything works end-to-end

Usage:
    python launch_experiment.py configs/base_config.yaml
"""

import sys
import rware
from config_loader import load_config, build_fleet
from heterogeneous_env import build_heterogeneous_env


def fleet_to_agents_cfg(fleet):
    """
    Converts our internal fleet format:
        [{"agent_id": 0, "speed": 0.9, "load_capacity": 3, "battery": 90}, ...]
    into the format heterogeneous_env.py expects:
        [{"speed": 0.9, "capacity": 3, "battery_capacity": 90}, ...]
    """
    agents_cfg = []
    for agent in fleet:
        agents_cfg.append({
            "speed": agent["speed"],
            "capacity": agent["load_capacity"],
            "battery_capacity": agent["battery"],
        })
    return agents_cfg


def launch(config_path):
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Build our fleet (uniform or explicit mode)
    fleet = build_fleet(config)
    print("\nGenerated fleet:")
    for agent in fleet:
        print(agent)

    # Convert to the format the environment wrapper expects,
    # and inject it into the config under heterogeneity.agents
    config["heterogeneity"]["agents"] = fleet_to_agents_cfg(fleet)

    # Build the real heterogeneous environment
    env = build_heterogeneous_env(config)
    print("\nEnvironment built successfully!")

    # Run a short test episode
    obs, info = env.reset()
    print(f"\nInitial observation received (showing structure only).")

    episode_length = config["environment"].get("episode_length", 10)
    test_steps = min(5, episode_length)  # just a quick sanity check, not the full episode

    for step in range(test_steps):
        actions = [0] * len(fleet)  # dummy actions, agent 0 = move action for now
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"Step {step + 1}: rewards={rewards}")

        if terminated or truncated:
            print("Episode ended early.")
            break

    print("\nTest run complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python launch_experiment.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    launch(config_path)