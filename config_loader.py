import random
import yaml
from mock_env import MockEnv


def load_config(config_path):
    """
    Reads a YAML config file and returns it as a dictionary.
    Example: config = load_config("configs/base_config.yaml")
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_fleet(config):
    """
    Builds a list of per-agent parameter dictionaries based on the
    heterogeneity settings in the config.

    Returns a list like:
    [{"agent_id": 0, "speed": 1.1, "load_capacity": 3, "battery": 80}, ...]
    """
    het_config = config.get("heterogeneity", {})
    num_agents = config["environment"]["num_agents"]

    # If heterogeneity isn't enabled, every agent gets the same defaults
    if not het_config.get("enabled", False):
        return [
            {"agent_id": i, "speed": 1.0, "load_capacity": 3, "battery": 100}
            for i in range(num_agents)
        ]

    mode = het_config.get("mode", "uniform")

    if mode == "explicit":
        fleet = het_config.get("explicit_fleet", [])
        if len(fleet) != num_agents:
            raise ValueError(
                f"explicit_fleet has {len(fleet)} agents but num_agents is {num_agents}"
            )
        return fleet

    elif mode == "uniform":
        ranges = het_config.get("uniform_ranges", {})
        speed_range = ranges.get("speed", [1.0, 1.0])
        capacity_range = ranges.get("load_capacity", [3, 3])
        battery_range = ranges.get("battery", [100, 100])

        fleet = []
        for i in range(num_agents):
            fleet.append({
                "agent_id": i,
                "speed": round(random.uniform(*speed_range), 2),
                "load_capacity": random.randint(int(capacity_range[0]), int(capacity_range[1])),
                "battery": random.randint(int(battery_range[0]), int(battery_range[1])),
            })
        return fleet

    else:
        raise ValueError(f"Unknown heterogeneity mode: {mode}")


if __name__ == "__main__":
    # Quick test with the base (uniform) config
    config = load_config("configs/base_config.yaml")
    fleet = build_fleet(config)

    print("Loaded config:")
    print(config)
    print("\nGenerated fleet:")
    for agent in fleet:
        print(agent)

    env = MockEnv(config=config)
    print(f"\nEnvironment initialized with grid_size={env.grid_size}, "
          f"num_agents={env.num_agents}, max_steps={env.max_steps}")