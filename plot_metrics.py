import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read the CSV file
    df = pd.read_csv('metrics.csv')
    
    # Create a figure with 3 subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Total Reward
    ax1.plot(df['Episode'], df['Total Reward'], marker='o', color='b')
    ax1.set_title('Total Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot Steps
    ax2.plot(df['Episode'], df['Steps'], marker='s', color='g')
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    # Plot Tasks Completed
    ax3.plot(df['Episode'], df['Tasks Completed'], marker='^', color='r')
    ax3.set_title('Tasks Completed per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Tasks')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = 'metrics_plot.png'
    plt.savefig(output_filename)
    print(f"Successfully saved metrics plot to {output_filename}")

if __name__ == '__main__':
    main()
