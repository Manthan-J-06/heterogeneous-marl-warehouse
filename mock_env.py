import random

class MockEnv:
    def __init__(self):
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return [0.0, 0.0], {}

    def step(self, action):
        self.current_step += 1
        
        # Random rewards for 2 agents
        rewards = [random.uniform(-1, 1), random.uniform(-1, 1)]
        
        # 10% chance to complete a task
        task_completed = random.random() < 0.1
        
        # Try to match typical Gymnasium API
        terminated = random.random() < 0.02
        truncated = self.current_step >= self.max_steps
        
        info = {'task_completed': task_completed}
        
        return [random.random(), random.random()], rewards, terminated, truncated, info
