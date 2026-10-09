import argparse
import json
import numpy as np
import random
import yaml
import sys
import torch
import os

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
    
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Checkpoint dir (default: runs/heteropolicy_<aware|blind>_<config stem>/checkpoints)")
    parser.add_argument("--checkpoint_interval", type=int, default=10000,
                        help="Save a checkpoint every N env steps (0 = final only)")
    parser.add_argument("--fleet_seed", type=int, default=None,
                        help="Seed for fleet sampling (pins the fleet drawn by build_fleet)")

    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    
    # parse the boolean flag robustly
    het_aware = str(args.heterogeneity_aware).lower() in ("true", "1", "yes", "t", "y")
    
    # Load configuration
    config = load_config(args.config)
    
    # Build standard fleet variables based on config
    if args.fleet_seed is not None:
        random.seed(args.fleet_seed)
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
    
    tag = "aware" if het_aware else "blind"
    cfg_stem = os.path.splitext(os.path.basename(args.config))[0]
    ckpt_dir = args.ckpt_dir or os.path.join("runs", f"heteropolicy_{tag}_{cfg_stem}", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "fleet.json"), "w") as _ff:
        json.dump({"speeds": list(env.speeds), "capacities": list(env.capacities),
                   "batteries": list(env.battery_capacities), "aware": het_aware,
                   "fleet_seed": args.fleet_seed}, _ff)
    last_ckpt_step = 0

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
        if args.checkpoint_interval and total_steps - last_ckpt_step >= args.checkpoint_interval:
            ckpt_path = os.path.join(ckpt_dir, f"heteropolicy_{tag}_step{total_steps}.pt")
            trainer.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
            last_ckpt_step = total_steps
        
        mean_agent_reward = episode_reward / num_agents
        print(f"Episode {episode_num:4d} | Steps: {total_steps:5d}/{MAX_ENV_STEPS} | "
              f"Mean Agent Reward: {mean_agent_reward:8.3f} | " 
              f"Actor Loss: {a_loss:8.4f} | Critic Loss: {c_loss:8.4f}")

    final_path = os.path.join(ckpt_dir, f"heteropolicy_{tag}_final.pt")
    trainer.save(final_path)
    print(f"  Saved final checkpoint: {final_path}")
    print("\nTraining completed.")

if __name__ == "__main__":
    main()
