"""Evaluate a trained HeteroPolicyTrainer checkpoint (or random actions) on a pinned fleet."""
import argparse, random
import numpy as np
from config_loader import load_config, build_fleet
from launch_experiment import fleet_to_agents_cfg
import heterogeneous_env as he
from algorithms.heterogeneity_aware_policy import HeteroPolicyTrainer

def evaluate(config_path, fleet_seed, episodes, ckpt=None, deterministic=False, base_seed=1000):
    random.seed(fleet_seed)
    c = load_config(config_path)
    fleet = build_fleet(c)
    c["heterogeneity"]["agents"] = fleet_to_agents_cfg(fleet)
    env = he.build_heterogeneous_env(c)
    n = len(env.action_space)
    policy = HeteroPolicyTrainer.load(ckpt) if ckpt else None
    label = "trained" if ckpt else "random"
    deliveries, ep_with, energy = [], 0, 0.0
    for ep in range(episodes):
        s = base_seed + ep
        random.seed(s); np.random.seed(s)
        obs, _ = env.reset(seed=s)
        d = 0
        for t in range(500):
            if policy:
                actions = policy.act(obs, fleet, deterministic=deterministic)
            else:
                actions = [random.randint(0, 4) for _ in range(n)]
            obs, r, term, trunc, info = env.step(actions)
            d += info["deliveries"]
            if term or trunc: break
        deliveries.append(d); ep_with += d > 0
        energy += sum(env.get_energy_consumed())
    tot = sum(deliveries)
    return {"label": label, "ckpt": ckpt, "fleet_seed": fleet_seed, "episodes": episodes,
            "total_deliveries": tot, "episodes_with_delivery": ep_with,
            "energy_per_delivery": (energy / tot) if tot else float("nan"),
            "fleet_speeds": env.speeds, "fleet_batteries": env.battery_capacities}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/low_variance_fleet.yaml")
    p.add_argument("--fleet_seed", type=int, default=7)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    print(evaluate(a.config, a.fleet_seed, a.episodes, a.ckpt, a.deterministic))
