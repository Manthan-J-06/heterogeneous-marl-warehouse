import matplotlib.pyplot as plt
import numpy as np

def generate_dashboard(metrics_logger, filepath="evaluation_summary.png"):
    """
    Generates a 4-subplot matplotlib dashboard evaluating performance 
    from a populated MetricsLogger instance.
    """
    history = metrics_logger.history
    if not history:
        print("Warning: MetricsLogger history is empty, bypassing dashboard generation.")
        return

    num_agents = metrics_logger.num_agents
    episodes = np.arange(1, len(history) + 1)
    
    tasks_completed = [ep['tasks_completed'] for ep in history]
    episode_lengths = [ep['steps'] for ep in history]
    
    # Pre-compute metrics
    energy_efficiency = []
    for ep in history:
        # Sum of 'per_agent_energy' for this episode
        total_energy = sum(ep.get('per_agent_energy', [0.0] * num_agents))
        tasks = ep['tasks_completed']
        
        if tasks > 0:
            energy_efficiency.append(total_energy / tasks)
        else:
            # Skip episodes with 0 tasks completed to avoid div by zero
            # We use np.nan to maintain array length for plotting without plotting the skips
            energy_efficiency.append(np.nan)
            
    # Calculate average per-agent reward across all episodes
    avg_per_agent_reward = [0.0] * num_agents
    for ep in history:
        for i in range(num_agents):
            avg_per_agent_reward[i] += ep['per_agent_rewards'][i]
            
    if len(history) > 0:
        avg_per_agent_reward = [r / len(history) for r in avg_per_agent_reward]

    # Create Subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Heterogeneous Multi-Agent Evaluation Summary', fontsize=16)

    # 1. Throughput
    axs[0, 0].plot(episodes, tasks_completed, marker='o', color='tab:blue')
    axs[0, 0].set_title('Throughput')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Tasks Completed')
    axs[0, 0].grid(True)

    # 2. Energy Efficiency
    axs[0, 1].plot(episodes, energy_efficiency, marker='s', color='tab:green')
    if all(np.isnan(v) for v in energy_efficiency):
        axs[0, 1].text(0.5, 0.5, 'No completed tasks in this run —\nenergy-per-task undefined',
                       horizontalalignment='center', verticalalignment='center',
                       transform=axs[0, 1].transAxes, fontsize=12, color='tab:red')
    axs[0, 1].set_title('Energy Efficiency')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Energy Per Task')
    axs[0, 1].grid(True)

    # 3. Task Completion Time (Episode Lengths)
    axs[1, 0].hist(episode_lengths, bins=10, color='tab:orange', edgecolor='black')
    axs[1, 0].set_title('Task Completion Time')
    axs[1, 0].set_xlabel('Steps to Completion / Truncation')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True, axis='y')

    # 4. Per-Agent Workload (Avg Reward)
    agents_x = np.arange(num_agents)
    bars = axs[1, 1].bar(agents_x, avg_per_agent_reward, color='tab:purple')
    axs[1, 1].set_title('Per-Agent Workload (Avg Reward)')
    axs[1, 1].set_xlabel('Agent ID')
    axs[1, 1].set_ylabel('Average Reward Contribution')
    axs[1, 1].set_xticks(agents_x)
    axs[1, 1].set_xticklabels([f"Agent {i}" for i in agents_x])
    axs[1, 1].grid(True, axis='y')

    # Add numeric labels to bars
    for bar in bars:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}",
                       ha='center', va='bottom', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath)
    print(f"Saved evaluation dashboard to {filepath}")
