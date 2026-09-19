from metrics_logger import MetricsLogger
from mock_env import MockEnv
from torch.utils.tensorboard import SummaryWriter

def main():
    logger = MetricsLogger()
    env = MockEnv()
    writer = SummaryWriter('runs/')
    
    num_episodes = 10
    
    for ep in range(num_episodes):
        logger.start_episode()
        obs, info = env.reset()
        
        done = False
        while not done:
            action = 0  # dummy action
            obs, reward, terminated, truncated, info = env.step(action)
            
            logger.log_step(reward, info.get('task_completed', False))
            
            if terminated or truncated:
                done = True
                
        logger.end_episode()
        
        # Log to TensorBoard
        ep_data = logger.history[-1]
        writer.add_scalar('Episode/Total_Reward', ep_data['total_reward'], ep)
        writer.add_scalar('Episode/Length', ep_data['steps'], ep)
        writer.add_scalar('Episode/Tasks_Completed', ep_data['tasks_completed'], ep)
        
    writer.close()
        
    print(f"{'Episode':<10} | {'Total Reward':<15} | {'Steps':<10} | {'Tasks Completed':<15}")
    print("-" * 57)
    for i, ep_data in enumerate(logger.history):
        print(f"{i+1:<10} | {ep_data['total_reward']:<15.2f} | {ep_data['steps']:<10} | {ep_data['tasks_completed']:<15}")

if __name__ == '__main__':
    main()
