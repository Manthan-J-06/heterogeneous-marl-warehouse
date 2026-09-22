"""
Config-driven QMIX training on the heterogeneous RWARE fleet.

Usage:
    # 20k step validation
    python train_qmix.py --config configs/low_variance_fleet.yaml --total-steps 20000
    
    # 500k step full run
    python train_qmix.py --config configs/low_variance_fleet.yaml --total-steps 500000 --checkpoint-interval 25000
"""

import argparse
import os
import time

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

# Swapped homogeneous wrapper for the heterogeneous environment builder
from heterogeneous_env import build_heterogeneous_env
from algorithms.qmix import QMIXTrainer


class MetricsLogger:
    """Minimal stand-in — replace with the existing Phase 1 MetricsLogger if available."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


def linear_epsilon(step, cfg):
    frac = min(1.0, step / cfg["epsilon_decay_steps"])
    return cfg["epsilon_start"] + frac * (cfg["epsilon_end"] - cfg["epsilon_start"])


def main(config_path, total_steps_override, checkpoint_interval_override):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override config values with CLI arguments if provided for Phase 3 scaling
    if total_steps_override is not None:
        cfg["total_env_steps"] = total_steps_override
    if checkpoint_interval_override is not None:
        cfg["checkpoint_interval"] = checkpoint_interval_override

    np.random.seed(cfg.get("seed", 42))

    # Phase 3: Initialize the heterogeneous environment
    env = build_heterogeneous_env(config_path=config_path, seed=cfg.get("seed", 42))
    state_dim = env.obs_dim * env.n_agents
    
    # Note: Ensure Huber loss, lower LR, and Double-Q flags are set in your YAML configs 
    # so they are correctly passed to QMIXTrainer here.
    trainer = QMIXTrainer(env.obs_dim, state_dim, env.n_agents, env.n_actions, cfg)
    
    # Create dynamic log directory based on the config name to prevent overwriting
    config_name = os.path.basename(config_path).replace(".yaml", "")
    run_log_dir = os.path.join(cfg.get("log_dir", "runs"), f"qmix_{config_name}")
    logger = MetricsLogger(run_log_dir)
    ckpt_dir = os.path.join(run_log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_len = 0
    episode_count = 0
    start_time = time.time()

    for step in range(1, cfg["total_env_steps"] + 1):
        epsilon = linear_epsilon(step, cfg)
        state = env.global_state(obs)
        actions = trainer.act(obs, epsilon)
        next_obs, reward, done, info = env.step(actions)
        next_state = env.global_state(next_obs)

        trainer.store(obs, actions, reward, next_obs, done, state, next_state)
        obs = next_obs
        episode_reward += float(np.sum(reward))
        episode_len += 1

        # Optimization step (Huber loss and grad clipping should be handled inside here)
        loss = trainer.train_step(cfg["batch_size"])

        if done:
            logger.log_scalar("train/episode_reward", episode_reward, episode_count)
            logger.log_scalar("train/episode_length", episode_len, episode_count)
            episode_count += 1
            episode_reward = 0.0
            episode_len = 0
            obs, _ = env.reset()

        if step % cfg.get("log_interval", 1000) == 0:
            elapsed = time.time() - start_time
            logger.log_scalar("train/epsilon", epsilon, step)
            if loss is not None:
                logger.log_scalar("train/loss", loss, step)
            print(f"[step {step}/{cfg['total_env_steps']}] "
                  f"episodes={episode_count} epsilon={epsilon:.3f} "
                  f"loss={loss if loss is not None else float('nan'):.4f} "
                  f"elapsed={elapsed:.0f}s")

        if step % cfg["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"qmix_step{step}.pt")
            trainer.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    trainer.save(os.path.join(ckpt_dir, "qmix_final.pt"))
    logger.close()
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to heterogeneous fleet YAML config")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total environment steps")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override checkpoint save interval")
    args = parser.parse_args()
    
    main(args.config, args.total_steps, args.checkpoint_interval)