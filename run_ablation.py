"""
run_ablation.py
----------------
Ablation experiment runner.

Given a list of config files, this script runs each one for a fixed
number of episodes, logs metrics using Manthan's MetricsLogger, and
combines results across all configs into one comparison table.

Usage:
    python run_ablation.py configs/low_variance_fleet.yaml configs/high_variance_fleet.yaml
"""

import sys
import csv
import random

import rware  # registers RWARE environments with gymnasium
from config_loader import load_config, build_fleet
from heterogeneous_env import build_heterogeneous_env
from metrics_logger import MetricsLogger


def fleet_to_agents_cfg(fleet):
    """Converts our fleet format into the format heterogeneous_env.py expects."""
    return [
        {
            "speed": agent["speed"],
            "capacity": agent["load_capacity"],
            "battery_capacity": agent["battery"],
        }
        for agent in fleet
    ]


def run_single_config(config_path, num_episodes=5):
    """
    Runs one config for a fixed number of episodes, returns a MetricsLogger
    populated with results, plus the config dict used.
    """
    print(f"\n=== Running config: {config_path} ===")

    config = load_config(config_path)
    fleet = build_fleet(config)
    config["heterogeneity"]["agents"] = fleet_to_agents_cfg(fleet)

    env = build_heterogeneous_env(config)
    num_agents = config["environment"]["num_agents"]
    episode_length = config["environment"].get("episode_length", 50)

    logger = MetricsLogger(num_agents=num_agents)

    for ep in range(num_episodes):
        logger.start_episode()
        obs, info = env.reset()

        for step in range(episode_length):
            actions = [random.randint(0, 4) for _ in range(num_agents)]
            obs, rewards, terminated, truncated, info = env.step(actions)

            task_completed = info.get("task_completed", False)
            logger.log_step(list(rewards), task_completed)

            if terminated or truncated:
                break

        logger.end_episode(env=env)
        print(f"  Episode {ep + 1}/{num_episodes} done "
              f"({logger.history[-1]['steps']} steps, "
              f"{logger.history[-1]['tasks_completed']} tasks completed)")

    return logger, config


def summarize_run(config_path, logger):
    """Computes summary stats across all episodes for one config."""
    history = logger.history
    n_episodes = len(history)

    avg_tasks = sum(ep["tasks_completed"] for ep in history) / n_episodes
    avg_steps = sum(ep["steps"] for ep in history) / n_episodes

    total_energy = sum(sum(ep.get("per_agent_energy", [])) for ep in history)
    total_tasks = sum(ep["tasks_completed"] for ep in history)
    energy_per_task = (total_energy / total_tasks) if total_tasks > 0 else float("nan")

    # Per-agent average reward (workload indicator)
    num_agents = logger.num_agents
    per_agent_avg = [0.0] * num_agents
    for ep in history:
        for i in range(num_agents):
            per_agent_avg[i] += ep["per_agent_rewards"][i]
    per_agent_avg = [float(r / n_episodes) for r in per_agent_avg]

    return {
        "config": config_path,
        "avg_throughput": round(avg_tasks, 2),
        "avg_energy_per_task": round(energy_per_task, 2) if total_tasks > 0 else "N/A",
        "avg_episode_length": round(avg_steps, 2),
        "per_agent_avg_reward": [round(r, 2) for r in per_agent_avg],
    }


def save_combined_results(summaries, filepath="ablation_results.csv"):
    """Saves one combined CSV comparing all configs."""
    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Config", "Avg Throughput", "Avg Energy Per Task",
            "Avg Episode Length", "Per-Agent Avg Reward"
        ])
        for s in summaries:
            writer.writerow([
                s["config"],
                s["avg_throughput"],
                s["avg_energy_per_task"],
                s["avg_episode_length"],
                s["per_agent_avg_reward"],
            ])
    print(f"\nCombined results saved to {filepath}")


def print_comparison_table(summaries):
    """Prints a simple side-by-side comparison table to the console."""
    print("\n=== Ablation Comparison ===")
    print(f"{'Config':<35} {'Throughput':<12} {'Energy/Task':<14} {'Ep Length':<12}")
    print("-" * 75)
    for s in summaries:
        print(f"{s['config']:<35} {s['avg_throughput']:<12} "
              f"{s['avg_energy_per_task']:<14} {s['avg_episode_length']:<12}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ablation.py <config1.yaml> <config2.yaml> ...")
        sys.exit(1)

    config_paths = sys.argv[1:]
    summaries = []

    for config_path in config_paths:
        logger, config = run_single_config(config_path, num_episodes=5)
        summary = summarize_run(config_path, logger)
        summaries.append(summary)

    print_comparison_table(summaries)
    save_combined_results(summaries)