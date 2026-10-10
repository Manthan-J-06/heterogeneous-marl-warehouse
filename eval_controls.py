"""Biased-random control policies on a pinned fleet. Compare against trained-policy deliveries.
Usage: python eval_controls.py --fleet_seed 1 [--config configs/low_variance_fleet.yaml]
"""
import argparse, random
import numpy as np
from config_loader import load_config, build_fleet
from launch_experiment import fleet_to_agents_cfg
import heterogeneous_env as he

CONTROLS = [("uniform", [.2] * 5), ("fwd40", [.15, .4, .15, .15, .15]), ("left40", [.15, .15, .4, .15, .15])]

def run_controls(config_path, fleet_seed, base_seeds=(1000, 2000, 3000, 4000), episodes=50):
    random.seed(fleet_seed)
    c = load_config(config_path)
    c["heterogeneity"]["agents"] = fleet_to_agents_cfg(build_fleet(c))
    env = he.build_heterogeneous_env(c)
    n = env.n_agents
    out = {}
    for name, p in CONTROLS:
        tot = 0
        for b in base_seeds:
            for ep in range(episodes):
                s = b + ep
                random.seed(s); np.random.seed(s); env.reset(seed=s)
                for t in range(500):
                    _, r, term, trunc, info = env.step(list(np.random.choice(5, size=n, p=p)))
                    tot += info["deliveries"]
                    if term or trunc:
                        break
        out[name] = tot
    return env, out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/low_variance_fleet.yaml")
    ap.add_argument("--fleet_seed", type=int, required=True)
    a = ap.parse_args()
    env, res = run_controls(a.config, a.fleet_seed)
    print("fleet speeds:", env.speeds, "| capacities:", env.capacities, "| batteries:", env.battery_capacities)
    for k, v in res.items():
        print(k, "| deliveries over 200 episodes:", v)
