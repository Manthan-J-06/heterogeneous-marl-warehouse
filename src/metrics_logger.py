"""
metrics_logger.py — Handles logging for Multi-Agent Reinforcement Learning (MARL) experiments.

Provides the `MetricsLogger` class, which tracks per-agent rewards as well as the
aggregated team reward. It expects a list of rewards from the multi-agent environment step.
"""

from typing import List, Sequence
import numpy as np


class MetricsLogger:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        
        # Track totals for the current episode
        self.episode_team_reward = 0.0
        self.episode_agent_rewards = [0.0] * num_agents
        
        self.episode_steps = 0

    def reset_episode(self):
        """Reset the running totals at the start of a new episode."""
        self.episode_team_reward = 0.0
        self.episode_agent_rewards = [0.0] * self.num_agents
        self.episode_steps = 0

    def log_step(self, rewards: Sequence[float]):
        """
        Record the rewards for a single environment step.
        
        Args:
            rewards: A list/array of floats representing the reward for each agent.
                     e.g., [0.0, 1.0] for a 2-agent environment.
        """
        assert len(rewards) == self.num_agents, (
            f"Expected {self.num_agents} rewards, but got {len(rewards)}."
        )
        
        self.episode_steps += 1
        
        # 1. Update total team reward
        self.episode_team_reward += sum(rewards)
        
        # 2. Update individual agent rewards
        for agent_id, reward in enumerate(rewards):
            self.episode_agent_rewards[agent_id] += float(reward)

    def get_episode_summary(self) -> dict:
        """
        Returns a dictionary summarizing the episode's metrics.
        Useful for printing or logging to TensorBoard / Weights & Biases.
        """
        summary = {
            "steps": self.episode_steps,
            "team_reward": self.episode_team_reward,
        }
        
        for agent_id, total_reward in enumerate(self.episode_agent_rewards):
            summary[f"agent_{agent_id}_reward"] = total_reward
            
        return summary

    def print_summary(self, episode: int):
        """Utility method to print the episode summary to the console."""
        summary = self.get_episode_summary()
        print(f"--- Episode {episode} Summary ---")
        print(f"  Steps:       {summary['steps']}")
        print(f"  Team Reward: {summary['team_reward']:.2f}")
        for agent_id in range(self.num_agents):
            print(f"  Agent {agent_id}:     {summary[f'agent_{agent_id}_reward']:.2f}")
        print("-" * 31)

