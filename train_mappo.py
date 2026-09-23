"""
Config-driven MAPPO training on the heterogeneous RWARE fleet.

Usage:
    python train_mappo.py --config configs/low_variance_fleet.yaml

See train_qmix.py's module docstring for the note on swapping in the
existing Phase 1 MetricsLogger instead of the minimal one defined here.
"""

import argparse
import os
import time

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from heterogeneous_env import build_heterogeneous_env
from algorithms.mappo import MAPPOTrainer
from config_loader import build_fleet
from launch_experiment import fleet_to_agents_cfg


class MetricsLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg.get("seed", 42))

    fleet = build_fleet(cfg)
    if "heterogeneity" not in cfg:
        cfg["heterogeneity"] = {}
    cfg["heterogeneity"]["agents"] = fleet_to_agents_cfg(fleet)
    env = build_heterogeneous_env(cfg)
    print(f"Speeds: {env.speeds}, Capacities: {env.capacities}, Battery capacities: {env.battery_capacities}")

    state_dim = env.obs_dim * env.n_agents
    trainer = MAPPOTrainer(env.obs_dim, state_dim, env.n_agents, env.n_actions, cfg)

    config_name = os.path.basename(config_path).replace(".yaml", "")
    run_log_dir = os.path.join(cfg.get("log_dir", "runs"), f"mappo_{config_name}")
    logger = MetricsLogger(run_log_dir)
    ckpt_dir = os.path.join(run_log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_len = 0
    episode_count = 0
    update_count = 0
    start_time = time.time()

    total_updates = cfg["total_env_steps"] // cfg["rollout_len"]

    for update in range(1, total_updates + 1):
        for _ in range(cfg["rollout_len"]):
            state = env.global_state(obs)
            actions, logprobs, value = trainer.act(obs, state)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            if isinstance(terminated, (list, tuple, np.ndarray)):
                done = all(terminated) or all(truncated)
            else:
                done = bool(terminated) or bool(truncated)
            trainer.buffer.add(obs, state, actions, logprobs, reward, done, value)
            obs = next_obs
            episode_reward += float(np.sum(reward))
            episode_len += 1

            if done:
                logger.log_scalar("train/episode_reward", episode_reward, episode_count)
                logger.log_scalar("train/episode_length", episode_len, episode_count)
                episode_count += 1
                episode_reward = 0.0
                episode_len = 0
                obs, _ = env.reset()

        last_state = env.global_state(obs)
        stats = trainer.update(last_state)
        update_count += 1
        step = update * cfg["rollout_len"]

        if update % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            for k, v in stats.items():
                logger.log_scalar(f"train/{k}", v, step)
            print(f"[update {update}/{total_updates}, step {step}] "
                  f"episodes={episode_count} policy_loss={stats['policy_loss']:.4f} "
                  f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} "
                  f"elapsed={elapsed:.0f}s")

        if step % cfg["checkpoint_interval"] < cfg["rollout_len"]:
            ckpt_path = os.path.join(ckpt_dir, f"mappo_step{step}.pt")
            trainer.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    trainer.save(os.path.join(ckpt_dir, "mappo_final.pt"))
    logger.close()
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mappo_rware_homogeneous.yaml")
    args = parser.parse_args()
    main(args.config)
