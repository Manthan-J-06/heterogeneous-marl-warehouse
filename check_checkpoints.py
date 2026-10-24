"""Re-evaluate archived MAPPO checkpoints with extra diagnostics (no retraining).

For each training seed it plays the standard 200 test episodes (base seeds 1000..4000 x 50)
and reports total deliveries and energy per delivery. Aware checkpoints are evaluated three ways:
  real    - the robots' true [speed, capacity/10, battery/1000]
  zero    - all-zero robot numbers
  shuffle - robot numbers rotated by one (robot i gets robot i+1's numbers)
If aware really used the numbers, zero/shuffle should score clearly lower than real.

Usage (from the repo root):
  python check_checkpoints.py                       # seeds 11 12 13, fleet 2, low_variance_fleet
  python check_checkpoints.py --seeds 11 --episodes 50 --base_seeds 1000   # quick test
  python check_checkpoints.py --config configs/tight_battery_fleet.yaml --stem tight_battery_fleet --seeds 11 12 13

Results are also saved to logs/check_checkpoints_<stem>_fleet<N>.json
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from config_loader import load_config, build_fleet  # noqa: E402
from launch_experiment import fleet_to_agents_cfg  # noqa: E402
import heterogeneous_env as he  # noqa: E402
from algorithms.mappo import MAPPOTrainer  # noqa: E402


def make_env(config, fleet_seed):
    random.seed(fleet_seed)
    c = load_config(config)
    c["heterogeneity"]["agents"] = fleet_to_agents_cfg(build_fleet(c))
    return c, he.build_heterogeneous_env(c)


def evaluate(c, env, ckpt_dir, mode, base_seeds, episodes):
    meta = json.load(open(os.path.join(ckpt_dir, "fleet.json")))
    assert meta["speeds"] == list(env.speeds) and meta["capacities"] == list(env.capacities) \
        and meta["batteries"] == list(env.battery_capacities), "fleet mismatch vs fleet.json"
    n = env.n_agents
    real = [[sp, cp / 10.0, bt / 1000.0]
            for sp, cp, bt in zip(env.speeds, env.capacities, env.battery_capacities)]
    if not meta["aware"]:
        props = None
    elif mode == "real":
        props = real
    elif mode == "zero":
        props = [[0.0, 0.0, 0.0] for _ in range(n)]
    elif mode == "shuffle":
        props = real[1:] + real[:1]
    else:
        raise ValueError(mode)
    cfg = dict(c)
    cfg.update(gamma=.99, gae_lambda=.95, clip_eps=.2, entropy_coef=.01, value_loss_coef=.5, ppo_epochs=1)
    trainer = MAPPOTrainer(env.obs_dim, env.obs_dim * n, n, env.n_actions, cfg, agent_props=props)
    ck = torch.load(os.path.join(ckpt_dir, "mappo_final.pt"), map_location="cpu", weights_only=False)
    trainer.actor.load_state_dict(ck["actor"])
    trainer.critic.load_state_dict(ck["critic"])
    deliveries, energy = 0, 0.0
    for b in base_seeds:
        for ep in range(episodes):
            s = b + ep
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            obs, _ = env.reset(seed=s)
            for _ in range(500):
                obs = np.asarray(obs)
                actions, _, _ = trainer.act(obs, env.global_state(obs))
                obs, _, term, trunc, info = env.step(actions)
                deliveries += info["deliveries"]
                if term or trunc:
                    break
            energy += sum(env.get_energy_consumed())
    return deliveries, (energy / deliveries if deliveries else float("nan")), bool(meta["aware"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/low_variance_fleet.yaml")
    ap.add_argument("--stem", default="low_variance_fleet")
    ap.add_argument("--fleet_seed", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--base_seeds", type=int, nargs="+", default=[1000, 2000, 3000, 4000])
    ap.add_argument("--episodes", type=int, default=50)
    a = ap.parse_args()

    c, env = make_env(a.config, a.fleet_seed)
    print("fleet speeds", env.speeds, "capacities", env.capacities, "batteries", env.battery_capacities, flush=True)
    results = []
    for seed in a.seeds:
        for kind in ("blind", "aware"):
            d = os.path.join("archive", f"fleet{a.fleet_seed}", f"mappo_{kind}_{a.stem}_seed{seed}", "checkpoints")
            if not os.path.exists(d):
                print(f"missing {d}", flush=True)
                continue
            for mode in (("real",) if kind == "blind" else ("real", "zero", "shuffle")):
                dl, epd, _ = evaluate(c, env, d, mode, a.base_seeds, a.episodes)
                row = {"train_seed": seed, "policy": kind, "robot_numbers": mode if kind == "aware" else "n/a",
                       "deliveries": dl, "energy_per_delivery": round(epd, 1)}
                results.append(row)
                print(row, flush=True)
    os.makedirs("logs", exist_ok=True)
    out = f"logs/check_checkpoints_{a.stem}_fleet{a.fleet_seed}.json"
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out)

    print("\n| seed | policy | robot numbers | deliveries | energy per delivery |")
    print("| :--- | :--- | :--- | :---: | :---: |")
    for r in results:
        print(f"| {r['train_seed']} | {r['policy']} | {r['robot_numbers']} | {r['deliveries']} | {r['energy_per_delivery']} |")


if __name__ == "__main__":
    main()
