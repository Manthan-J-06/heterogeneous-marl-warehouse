"""Varying-fleet experiment: a NEW random fleet every episode.

With the fleet pinned, the robot ID already tells a policy which robot is which, so the
"aware" inputs were redundant. Here the fleet is resampled every episode, so the ID says
nothing about speed/capacity/battery -- only the aware inputs do. The aware policy also
sees each robot's CURRENT charge (fraction of battery left), not just its capacity.

Usage (repo root):
  python varying.py run blind 11 --steps 300000      # train + evaluate, saves logs/varying<steps>_blind_seed11.json
  python varying.py run aware 11 --steps 300000
  python varying.py control                          # random / fwd40 / left40 on the same test fleets
  python varying.py diag 11 --steps 300000           # aware checkpoint with real / zero / shuffled robot numbers
  python varying.py run blind 11 --steps 20000 --episodes 5 --base_seeds 1000   # quick smoke test

Test set: for every test episode s (base seed 1000/2000/3000/4000 + 0..49) the fleet is
drawn from random.Random(s + 1_000_000), so every policy and control sees identical fleets.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.getcwd())
import heterogeneous_env as he  # noqa: E402
from algorithms.mappo_dynamic import DynamicMAPPOTrainer  # noqa: E402
from fleet_sampling import sample_agents_cfg, get_props  # noqa: E402

CONFIG = "configs/varying_fleet.yaml"
BASE_SEEDS = [1000, 2000, 3000, 4000]
PROPS_DIM = 4
CONTROLS = {"uniform": [.2] * 5, "fwd40": [.15, .4, .15, .15, .15], "left40": [.15, .15, .4, .15, .15]}


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(cfg):
    c = dict(cfg)
    c["heterogeneity"] = dict(cfg["heterogeneity"])
    c["heterogeneity"]["agents"] = sample_agents_cfg(cfg, random.Random(0))  # placeholder; reset per episode
    return he.build_heterogeneous_env(c)


def stem_of(path):
    return os.path.basename(path).replace(".yaml", "")


def ckpt_dir_for(mode, seed, steps, config):
    return os.path.join("archive", "varying", f"{mode}_{stem_of(config)}_{steps}_seed{seed}")


# ---------------------------------------------------------------- training
def train(mode, seed, steps, config=CONFIG):
    cfg = load_cfg(config)
    aware = mode == "aware"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    fleet_rng = random.Random(seed * 1000 + 1)   # training fleets (test fleets use seeds >= 1,000,000)
    env = make_env(cfg)
    state_dim = env.obs_dim * env.n_agents
    tr = DynamicMAPPOTrainer(env.obs_dim, state_dim, env.n_agents, env.n_actions, cfg,
                             props_dim=PROPS_DIM if aware else 0)
    out = ckpt_dir_for(mode, seed, steps, config)
    os.makedirs(out, exist_ok=True)

    def new_episode(first=False):
        env.set_fleet(sample_agents_cfg(cfg, fleet_rng))
        return env.reset(seed=seed)[0] if first else env.reset()[0]

    obs = new_episode(first=True)
    rollout_len = cfg["rollout_len"]
    total_updates = steps // rollout_len
    episodes, t0 = 0, time.time()
    for upd in range(1, total_updates + 1):
        for _ in range(rollout_len):
            state = env.global_state(obs)
            props = get_props(env) if aware else None
            actions, logp, value = tr.act(obs, state, props, record=True)
            nxt, reward, term, trunc, _ = env.step(actions)
            done = (all(term) or all(trunc)) if isinstance(term, (list, tuple, np.ndarray)) else bool(term) or bool(trunc)
            tr.buffer.add(obs, state, actions, logp, reward, done, value)
            obs = nxt
            if done:
                episodes += 1
                obs = new_episode()
        stats = tr.update(env.global_state(obs), get_props(env) if aware else None)
        if upd % max(1, cfg["log_interval"] // rollout_len) == 0:
            print(f"[update {upd}/{total_updates}] episodes={episodes} policy_loss={stats['policy_loss']:.4f} "
                  f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
    tr.save(os.path.join(out, "mappo_final.pt"))
    json.dump({"aware": aware, "props_dim": PROPS_DIM if aware else 0, "seed": seed, "steps": steps,
               "config": config}, open(os.path.join(out, "meta.json"), "w"))
    print("trained ->", out, flush=True)
    return out


# -------------------------------------------------------------- evaluation
def evaluate(policy, config=CONFIG, ckpt=None, props_mode="real", base_seeds=BASE_SEEDS, episodes=50):
    """policy: 'mappo' (needs ckpt dir) or one of CONTROLS. Returns (deliveries, energy_per_delivery)."""
    cfg = load_cfg(config)
    env = make_env(cfg)
    n = env.n_agents
    tr, aware = None, False
    if policy == "mappo":
        meta = json.load(open(os.path.join(ckpt, "meta.json")))
        aware = meta["aware"]
        tr = DynamicMAPPOTrainer(env.obs_dim, env.obs_dim * n, n, env.n_actions, cfg, props_dim=meta["props_dim"])
        ck = torch.load(os.path.join(ckpt, "mappo_final.pt"), map_location="cpu", weights_only=False)
        tr.actor.load_state_dict(ck["actor"]); tr.critic.load_state_dict(ck["critic"])
    deliveries, energy = 0, 0.0
    for b in base_seeds:
        for ep in range(episodes):
            s = b + ep
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            env.set_fleet(sample_agents_cfg(cfg, random.Random(s + 1_000_000)))
            obs, _ = env.reset(seed=s)
            for _ in range(500):
                if tr is not None:
                    props = None
                    if aware:
                        props = np.array(get_props(env), dtype=np.float32)
                        if props_mode == "zero":
                            props = np.zeros_like(props)
                        elif props_mode == "shuffle":
                            props = np.roll(props, 1, axis=0)
                    actions, _, _ = tr.act(obs, env.global_state(obs), props)
                else:
                    actions = list(np.random.choice(5, size=n, p=CONTROLS[policy]))
                obs, _, term, trunc, info = env.step(actions)
                deliveries += info["deliveries"]
                if term or trunc:
                    break
            energy += sum(env.get_energy_consumed())
    return deliveries, (energy / deliveries if deliveries else float("nan"))


def save(path, obj):
    os.makedirs("logs", exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
    print("saved", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "control", "diag"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--base_seeds", type=int, nargs="+", default=BASE_SEEDS)
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()
    stem = stem_of(a.config)

    if a.cmd == "run":
        mode, seed = a.args[0], int(a.args[1])
        assert mode in ("blind", "aware")
        out = ckpt_dir_for(mode, seed, a.steps, a.config)
        if not a.eval_only:
            out = train(mode, seed, a.steps, a.config)
        tot, epd = evaluate("mappo", a.config, out, "real", a.base_seeds, a.episodes)
        print(f"{mode} seed {seed}: {tot} deliveries, {epd:.1f} energy/delivery", flush=True)
        save(f"logs/varying{a.steps}_{stem}_{mode}_seed{seed}.json",
             {"fleet_seed": 0, "mode": mode, "train_seed": seed, "steps": a.steps, "config": a.config,
              "total": tot, "energy_per_delivery": round(epd, 1)})
    elif a.cmd == "control":
        res = {}
        for name in CONTROLS:
            res[name], _ = evaluate(name, a.config, None, "real", a.base_seeds, a.episodes)
            print(name, res[name], flush=True)
        save(f"logs/varying_controls_{stem}.json", res)
        print(f"\nUse in build_long_rows.py:  {res['uniform']} {res['fwd40']} {res['left40']}")
    else:  # diag
        seed = int(a.args[0])
        out = ckpt_dir_for("aware", seed, a.steps, a.config)
        rows = []
        for pm in ("real", "zero", "shuffle"):
            tot, epd = evaluate("mappo", a.config, out, pm, a.base_seeds, a.episodes)
            rows.append({"train_seed": seed, "robot_numbers": pm, "deliveries": tot, "energy_per_delivery": round(epd, 1)})
            print(rows[-1], flush=True)
        save(f"logs/varying_diag_{stem}_seed{seed}.json", rows)


if __name__ == "__main__":
    main()
