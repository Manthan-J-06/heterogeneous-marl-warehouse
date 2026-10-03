"""Evaluate a trained HeteroPolicyTrainer checkpoint (or random actions) on a pinned fleet."""
import argparse, random, json, os
import numpy as np
from config_loader import load_config, build_fleet
from launch_experiment import fleet_to_agents_cfg
import heterogeneous_env as he
from algorithms.heterogeneity_aware_policy import HeteroPolicyTrainer
from algorithms.mappo import MAPPOTrainer
import torch

def evaluate(config_path, fleet_seed, episodes, ckpt=None, deterministic=False, base_seed=int(os.environ.get("BASE_SEED", 1000)), mappo_dir=None):
    random.seed(fleet_seed)
    c = load_config(config_path)
    fleet = build_fleet(c)
    c["heterogeneity"]["agents"] = fleet_to_agents_cfg(fleet)
    env = he.build_heterogeneous_env(c)
    n = len(env.action_space)
    policy = HeteroPolicyTrainer.load(ckpt) if ckpt else None
    label = "trained" if ckpt else "random"
    mappo = None
    if mappo_dir:
        meta = json.load(open(os.path.join(mappo_dir, "fleet.json")))
        assert meta["speeds"] == list(env.speeds) and meta["capacities"] == list(env.capacities) \
            and meta["batteries"] == list(env.battery_capacities), "fleet mismatch vs fleet.json"
        cfg = dict(c); cfg.update(gamma=.99, gae_lambda=.95, clip_eps=.2, entropy_coef=.01,
                                  value_loss_coef=.5, ppo_epochs=1)
        props = [[sp, cp / 10.0, bt / 1000.0] for sp, cp, bt in zip(env.speeds, env.capacities, env.battery_capacities)] if meta["aware"] else None
        mappo = MAPPOTrainer(env.obs_dim, env.obs_dim * n, n, env.n_actions, cfg, agent_props=props)
        ck = torch.load(os.path.join(mappo_dir, "mappo_final.pt"), map_location="cpu", weights_only=False)
        mappo.actor.load_state_dict(ck["actor"]); mappo.critic.load_state_dict(ck["critic"])
        label = "mappo_aware" if meta["aware"] else "mappo_blind"
    deliveries, ep_with, energy = [], 0, 0.0
    for ep in range(episodes):
        s = base_seed + ep
        random.seed(s); np.random.seed(s)
        obs, _ = env.reset(seed=s)
        d = 0
        for t in range(500):
            if mappo:
                actions, _, _ = mappo.act(obs, env.global_state(obs))
            elif policy:
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
    p.add_argument("--mappo_dir", default=None)
    a = p.parse_args()
    r = evaluate(a.config, a.fleet_seed, a.episodes, a.ckpt, a.deterministic, mappo_dir=a.mappo_dir)
    print(r['label'], r['fleet_seed'], 'deliveries:', r['total_deliveries'], 'episodes_with_delivery:', r['episodes_with_delivery'])
