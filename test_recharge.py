"""Verify the opt-in passive recharge in HeterogeneousRWAREWrapper.
Run from the repo root on a branch that has PR #18:  python test_recharge.py
"""
import os, random, tempfile
import numpy as np
import yaml
from config_loader import load_config, build_fleet
from launch_experiment import fleet_to_agents_cfg
import heterogeneous_env as he

CONFIG = "configs/low_variance_fleet.yaml"
NOOP, FORWARD = 0, 1


def make_env(fleet_seed=7, battery=None, speed=None, recharge=0.0):
    random.seed(fleet_seed)
    c = load_config(CONFIG)
    agents = fleet_to_agents_cfg(build_fleet(c))
    for a in agents:
        if battery is not None:
            a["battery_capacity"] = float(battery)
        if speed is not None:
            a["speed"] = float(speed)
    c["heterogeneity"]["agents"] = agents
    c["heterogeneity"]["recharge_rate"] = recharge
    return he.build_heterogeneous_env(c)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    return ok


def run_fixed(env, action, steps):
    env.reset(seed=0)
    for _ in range(steps):
        env.step([action] * env.n_agents)
    return env.get_battery_levels()


def test_config_passthrough():
    with open(CONFIG) as f:
        raw = yaml.safe_load(f)
    raw["heterogeneity"]["recharge_rate"] = 1.5
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as t:
        yaml.safe_dump(raw, t)
        path = t.name
    try:
        c = load_config(path)
        got = c["heterogeneity"].get("recharge_rate")
        env = he.build_heterogeneous_env(c)
        return check("YAML recharge_rate reaches the env", got == 1.5 and env.recharge_rate == 1.5,
                     f"config={got}, env={env.recharge_rate}")
    finally:
        os.remove(path)


def test_noop_balance():
    ok = True
    # speed 1.0, battery 10, all NOOP: idle cost 0.5/step, 4 steps.
    for rate, expect in [(0.0, 8.0), (0.25, 9.0), (0.5, 10.0), (2.0, 10.0)]:
        lv = run_fixed(make_env(battery=10, speed=1.0, recharge=rate), NOOP, 4)
        ok &= check(f"all-NOOP, recharge={rate}", all(abs(x - expect) < 1e-6 for x in lv),
                    f"expected {expect}, got {[round(x, 2) for x in lv]}")
    return ok


def test_move_does_not_recharge():
    lv = run_fixed(make_env(battery=100, speed=1.0, recharge=1.0), FORWARD, 5)
    return check("all-FORWARD does not recharge (cost 1.0/step)", all(abs(x - 95.0) < 1e-6 for x in lv),
                 f"expected 95.0, got {[round(x, 2) for x in lv]}")


def test_bounds():
    env = make_env(battery=5, recharge=4.0)
    caps = env.battery_capacities
    env.reset(seed=1)
    ok = True
    for _ in range(300):
        env.step(list(np.random.randint(0, 5, size=env.n_agents)))
        lv = env.get_battery_levels()
        ok &= all(-1e-9 <= l <= c + 1e-9 for l, c in zip(lv, caps))
    return check("battery stays within [0, capacity] under random actions", ok)


def sweep(battery=200, rates=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), episodes=5, steps=500):
    print(f"\nContrast sweep: battery={battery}, uniform random actions, {episodes} eps x {steps} steps")
    print("| recharge_rate | dead agent-steps | mean final battery / capacity |")
    print("| :---: | :---: | :---: |")
    for r in rates:
        env = make_env(battery=battery, recharge=r)
        dead = total = 0
        frac = []
        for s in range(episodes):
            random.seed(s); np.random.seed(s)
            env.reset(seed=s)
            for _ in range(steps):
                env.step(list(np.random.choice(5, size=env.n_agents)))
                lv = env.get_battery_levels()
                dead += sum(1 for x in lv if x <= 0.0)
                total += len(lv)
            frac.append(np.mean([x / c for x, c in zip(env.get_battery_levels(), env.battery_capacities)]))
        print(f"| {r} | {dead}/{total} ({100 * dead / total:.1f}%) | {np.mean(frac):.2f} |")


if __name__ == "__main__":
    results = [test_config_passthrough(), test_noop_balance(), test_move_does_not_recharge(), test_bounds()]
    sweep()
    print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED")
