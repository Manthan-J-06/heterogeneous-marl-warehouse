import yaml
import numpy as np
from pathlib import Path

from heterogeneous_env import build_heterogeneous_env
from metrics_logger import MetricsLogger
from evaluation_pipeline import generate_dashboard

# Ensure standard action-space behavior
try:
    import rware
except ImportError:
    pass

def main():
    config_path = "configs/heterogeneous_small.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override for this specific evaluation run
    if "heterogeneity" not in config:
        config["heterogeneity"] = {"enabled": True, "agents": []}
    
    config["heterogeneity"]["energy_weight"] = 0.05
    num_episodes = 15

    print(f"Building heterogeneous environment with {config_path}")
    env = build_heterogeneous_env(config)
    n_agents = len(env.speeds)

    logger = MetricsLogger(num_agents=n_agents)

    print(f"Running {num_episodes} evaluation episodes...")
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        logger.start_episode()
        
        episode_steps = 0
        while True:
            # Random policy sampling
            actions = [env.action_space[i].sample() for i in range(n_agents)]
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Tasks in rware give reward of 1.0. A task is completed if any reward is > 0.
            # In our wrapper, reward shrinks due to energy, so we check if base reward > eps.
            # But the simplest heuristic: if total adjusted reward > 0 or we want to look at info.
            # Let's just say a task completed if sum(rewards) > 0 (even with penalty, 1.0 - penalty usually > 0)
            task_completed = sum(rewards) > 0
            
            logger.log_step(rewards=rewards, task_completed=task_completed)
            episode_steps += 1
            
            if isinstance(terminated, (list, tuple, np.ndarray)):
                done_terminated = all(terminated)
            else:
                done_terminated = bool(terminated)

            if isinstance(truncated, (list, tuple, np.ndarray)):
                done_truncated = all(truncated)
            else:
                done_truncated = bool(truncated)

            if done_terminated or done_truncated or episode_steps >= 500:
                break
                
        # Inject energy states into the episode history using the new env support logic
        logger.end_episode(env=env)
        
        print(f"  Episode {ep:>2}/{num_episodes} completed in {episode_steps} steps.")

    env.close()
    
    logger.export_to_csv("evaluation_metrics.csv")
    print("\nMetrics exported to evaluation_metrics.csv")
    
    generate_dashboard(logger, "evaluation_summary.png")

if __name__ == "__main__":
    main()
