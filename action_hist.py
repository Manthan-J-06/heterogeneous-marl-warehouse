"""Print the action distribution and deliveries of archived MAPPO checkpoints.

Usage (from the repo root):
  python action_hist.py archive/fleet2/mappo_aware_low_variance_fleet_seed11/checkpoints [more dirs ...]
Each checkpoint dir must contain fleet.json and mappo_final.pt.
Actions: 0 NOOP, 1 FORWARD, 2 LEFT, 3 RIGHT, 4 TOGGLE_LOAD. Uniform = 0.20 each.
"""
import json, math, os, random, sys
import numpy as np
import torch
import yaml
from config_loader import build_fleet
from launch_experiment import fleet_to_agents_cfg
from heterogeneous_env import build_heterogeneous_env
from algorithms.mappo import MAPPOTrainer

CONFIG = "configs/low_variance_fleet.yaml"
EPISODES = 10
STEPS = 500
NAMES = ["NOOP", "FWD", "LEFT", "RIGHT", "TOGGLE"]


def load(ckpt_dir):
    meta = json.load(open(os.path.join(ckpt_dir, "fleet.json")))
    cfg = yaml.safe_load(open(CONFIG))
    random.seed(meta["fleet_seed"])
    fleet = build_fleet(cfg)
    cfg.setdefault("heterogeneity", {})["agents"] = fleet_to_agents_cfg(fleet)
    env = build_heterogeneous_env(cfg)
    props = None
    if meta["aware"]:
        props = [[sp, cp / 10.0, bt / 1000.0] for sp, cp, bt in zip(env.speeds, env.capacities, env.battery_capacities)]
    trainer = MAPPOTrainer(env.obs_dim, env.obs_dim * env.n_agents, env.n_agents, env.n_actions, cfg, agent_props=props)
    ck = torch.load(os.path.join(ckpt_dir, "mappo_final.pt"), map_location="cpu", weights_only=False)
    trainer.actor.load_state_dict(ck["actor"])
    trainer.critic.load_state_dict(ck["critic"])
    return env, trainer, meta


def main(dirs):
    print("| checkpoint | aware | deliveries (10 eps) | " + " | ".join(NAMES) + " | entropy of mean dist |")
    print("| :--- | :---: | :---: | " + " | ".join([":---:"] * len(NAMES)) + " | :---: |")
    for d in dirs:
        env, trainer, meta = load(d)
        torch.manual_seed(0)
        counts = np.zeros(env.n_actions)
        deliveries = 0
        for ep in range(EPISODES):
            s = 1000 + ep
            random.seed(s); np.random.seed(s)
            obs, _ = env.reset(seed=s)
            for _ in range(STEPS):
                actions, _, _ = trainer.act(obs, env.global_state(obs))
                for a in actions:
                    counts[a] += 1
                obs, r, term, trunc, info = env.step(actions)
                deliveries += info["deliveries"]
                if term or trunc:
                    break
        p = counts / counts.sum()
        ent = -sum(x * math.log(x) for x in p if x > 0)
        name = os.path.basename(os.path.dirname(os.path.normpath(d)))
        print(f"| {name} | {meta['aware']} | {deliveries} | " + " | ".join(f"{x:.2f}" for x in p) + f" | {ent:.2f} |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
