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


if __name__ == "__main__":
    # Load base config
    config = load_config("configs/base_config.yaml")
    print("Loaded config:")
    print(config)

    # Initialize the environment USING the config
    env = MockEnv(config=config)

    print(f"\nEnvironment initialized with:")
    print(f"Grid size: {env.grid_size}")
    print(f"Num agents: {env.num_agents}")
    print(f"Max steps: {env.max_steps}")

    # Quick test: reset the env and see the output
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")