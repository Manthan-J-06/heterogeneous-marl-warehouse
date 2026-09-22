"""
Config-driven QMIX training on the homogeneous RWARE fleet.

Usage:
    python train_qmix.py --config configs/qmix_rware_homogeneous.yaml

NOTE ON METRICS LOGGING:
This uses a minimal `MetricsLogger` wrapper around TensorBoard's
SummaryWriter. If Phase 1 already has a MetricsLogger class (per the PR
that added TensorBoard integration), swap the import below for that one —
the interface here (`log_scalar(tag, value, step)`) is deliberately tiny
so it should be a drop-in match or a two-line adapter at most.
"""

import argparse
import os
import time

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from envs.rware_wrapper import RwareEnvWrapper
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


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["seed"])

    env = RwareEnvWrapper(cfg["env_id"], seed=cfg["seed"])
    state_dim = env.obs_dim * env.n_agents
    trainer = QMIXTrainer(env.obs_dim, state_dim, env.n_agents, env.n_actions, cfg)
    logger = MetricsLogger(cfg["log_dir"])
    ckpt_dir = os.path.join(cfg["log_dir"], "checkpoints")
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

        loss = trainer.train_step(cfg["batch_size"])

        if done:
            logger.log_scalar("train/episode_reward", episode_reward, episode_count)
            logger.log_scalar("train/episode_length", episode_len, episode_count)
            episode_count += 1
            episode_reward = 0.0
            episode_len = 0
            obs, _ = env.reset()

        if step % cfg["log_interval"] == 0:
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
    parser.add_argument("--config", default="configs/qmix_rware_homogeneous.yaml")
    args = parser.parse_args()
    main(args.config)
