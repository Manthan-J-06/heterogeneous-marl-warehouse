import csv

class MetricsLogger:
    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents
        self.history = []
        self.current_episode = None

    def start_episode(self):
        self.current_episode = {
            'total_reward': 0.0,
            'per_agent_rewards': [0.0] * self.num_agents,
            'per_agent_energy': [0.0] * self.num_agents,
            'steps': 0,
            'tasks_completed': 0
        }

    def log_step(self, rewards: list, task_completed: bool):
        if self.current_episode is None:
            raise ValueError("Episode has not started. Call start_episode() first.")
        self.current_episode['total_reward'] += sum(rewards)
        for i, r in enumerate(rewards):
            if i < self.num_agents:
                self.current_episode['per_agent_rewards'][i] += r
        self.current_episode['steps'] += 1
        if task_completed:
            self.current_episode['tasks_completed'] += 1

    def end_episode(self, env=None):
        if self.current_episode is None:
            raise ValueError("No active episode to end.")
            
        if env is not None and hasattr(env, "get_energy_consumed"):
            self.current_episode['per_agent_energy'] = list(env.get_energy_consumed())
            
        self.history.append(self.current_episode)
        self.current_episode = None

    def export_to_csv(self, filepath: str):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Episode', 'Total Reward', 'Steps', 'Tasks Completed']
            for i in range(self.num_agents):
                headers.append(f'Agent_{i}_Reward')
            for i in range(self.num_agents):
                headers.append(f'Agent_{i}_Energy')
            writer.writerow(headers)
            for idx, ep_data in enumerate(self.history):
                row = [
                    idx + 1,
                    ep_data['total_reward'],
                    ep_data['steps'],
                    ep_data['tasks_completed']
                ]
                row.extend(ep_data['per_agent_rewards'])
                row.extend(ep_data.get('per_agent_energy', [0.0] * self.num_agents))
                writer.writerow(row)

