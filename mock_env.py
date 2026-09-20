import random

class MockEnv:
    def __init__(self, config=None):
        # If a config dict is passed in, use its values.
        # Otherwise fall back to sensible defaults.
        if config is not None:
            env_config = config["environment"]
            self.grid_size = env_config["grid_size"]
            self.num_agents = env_config["num_agents"]
            self.max_steps = env_config["episode_length"]
        else:
            self.grid_size = 10
            self.num_agents = 2
            self.max_steps = 100

        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return [0.0] * self.num_agents, {}

    def step(self, action):
        self.current_step += 1

        # Random rewards, one per agent
        rewards = [random.uniform(-1, 1) for _ in range(self.num_agents)]

        # 10% chance to complete a task
        task_completed = random.random() < 0.1

        # Try to match typical Gymnasium API
        terminated = random.random() < 0.02
        truncated = self.current_step >= self.max_steps

        info = {'task_completed': task_completed}

        return [random.random()] * self.num_agents, rewards, terminated, truncated, info
