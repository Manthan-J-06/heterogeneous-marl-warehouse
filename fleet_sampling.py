"""Sample a fresh fleet from a config's uniform_ranges, using its own RNG.

Uses a private random.Random so it never disturbs the global RNG that the
environment uses for speed no-ops.
"""
import random


def sample_agents_cfg(config, rng: random.Random):
    het = config["heterogeneity"]
    n = config["environment"]["num_agents"]
    r = het["uniform_ranges"]
    return [{
        "speed": round(rng.uniform(*r["speed"]), 2),
        "capacity": rng.randint(int(r["load_capacity"][0]), int(r["load_capacity"][1])),
        "battery_capacity": rng.randint(int(r["battery"][0]), int(r["battery"][1])),
    } for _ in range(n)]


def get_props(env):
    """Per-robot features: [speed, capacity/10, battery_capacity/1000, current_charge_fraction]."""
    levels = env.get_battery_levels()
    return [[sp, cp / 10.0, bc / 1000.0, (lv / bc) if bc > 0 else 0.0]
            for sp, cp, bc, lv in zip(env.speeds, env.capacities, env.battery_capacities, levels)]
