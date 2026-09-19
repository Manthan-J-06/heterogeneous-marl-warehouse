import csv

class MetricsLogger:
    def __init__(self):
        self.history = []
        self.current_episode = None

    def start_episode(self):
        self.current_episode = {
            'total_reward': 0.0,
            'steps': 0,
            'tasks_completed': 0
        }

    def log_step(self, reward: float, task_completed: bool):
        if self.current_episode is None:
            raise ValueError("Episode has not started. Call start_episode() first.")
        self.current_episode['total_reward'] += reward
        self.current_episode['steps'] += 1
        if task_completed:
            self.current_episode['tasks_completed'] += 1

    def end_episode(self):
        if self.current_episode is None:
            raise ValueError("No active episode to end.")
        self.history.append(self.current_episode)
        self.current_episode = None

    def export_to_csv(self, filepath: str):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total Reward', 'Steps', 'Tasks Completed'])
            for idx, ep_data in enumerate(self.history):
                writer.writerow([
                    idx + 1,
                    ep_data['total_reward'],
                    ep_data['steps'],
                    ep_data['tasks_completed']
                ])
