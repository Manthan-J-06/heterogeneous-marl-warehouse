import argparse
import numpy as np
import yaml
import sys
import torch

from config_loader import load_config, build_fleet
from launch_experiment import fleet_to_agents_cfg
from heterogeneous_env import build_heterogeneous_env
from algorithms.heterogeneity_aware_policy import HeteroPolicyTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/low_variance_fleet.yaml")
    parser.add_argument("--heterogeneity_aware", type=str, default="True", 
                        help="True/False flag or 1/0")
    parser.add_argument("--total_steps", type=int, default=5000,
                        help="Total environment steps for training")
    
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    
    # parse the boolean flag robustly
    het_aware = str(args.heterogeneity_aware).lower() in ("true", "1", "yes", "t", "y")
    
    # Load configuration
    config = load_config(args.config)
    
    # Build standard fleet variables based on config
    fleet = build_fleet(config)
    
    if "heterogeneity" not in config:
        config["heterogeneity"] = {}
    config["heterogeneity"]["agents"] = fleet_to_agents_cfg(fleet)

    # Initialize environment
    env = build_heterogeneous_env(config)
    
    print(f"env.speeds: {env.speeds}")
    print(f"env.capacities: {env.capacities}")
    print(f"env.battery_capacities: {env.battery_capacities}")
    
    # Environment specs
    num_agents = len(env.action_space)
    # the obs is a tuple of Box for each agent. Pick the first agent's observation shape as dim
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].n
    
    print(f"==========================================")
    print(f"Initializing HeteroPolicyTrainer")
    print(f"Agents: {num_agents}, Obs Dim: {obs_dim}, Act Dim: {act_dim}")
    print(f"Heterogeneity Aware: {het_aware}")
    print(f"==========================================\n")

    # Initialize trainer model
    trainer = HeteroPolicyTrainer(
        obs_dim=obs_dim,
        num_agents=num_agents,
        act_dim=act_dim,
        hidden_dim=32,
        learning_rate=1e-4,
        heterogeneity_aware=het_aware
    )
    
    MAX_ENV_STEPS = args.total_steps
    total_steps = 0
    episode_num = 0
    
    while total_steps < MAX_ENV_STEPS:
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done and total_steps < MAX_ENV_STEPS:
            # act
            actions = trainer.get_actions(obs, fleet)
            
            # step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            trainer.store_rewards(rewards)
            
            if episode_num == 0 and total_steps < 20:
                print(f"Raw reward step {total_steps}: {rewards}")
            
            # Sum up rewards from all agents for this step
            episode_reward += sum(rewards)
            total_steps += 1
            
            # Handle Gym generic terminations/truncations format
            if isinstance(terminations, (tuple, list)):
                term = any(terminations)
            else:
                term = terminations
                
            if isinstance(truncations, (tuple, list)):
                trunc = any(truncations)
            else:
                trunc = truncations
                
            done = term or trunc
            
            obs = next_obs
        
        # update policy at end of episode
        a_loss, c_loss = trainer.train_episode()
        episode_num += 1
        
        mean_agent_reward = episode_reward / num_agents
        print(f"Episode {episode_num:4d} | Steps: {total_steps:5d}/{MAX_ENV_STEPS} | "
              f"Mean Agent Reward: {mean_agent_reward:8.3f} | " 
              f"Actor Loss: {a_loss:8.4f} | Critic Loss: {c_loss:8.4f}")

    print("\nTraining completed.")

if __name__ == "__main__":
    main()
