import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class HeteroActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, num_agents, act_dim, hidden_dim=32, heterogeneity_aware=False):
        super().__init__()
        self.heterogeneity_aware = heterogeneity_aware
        
        # input is obs + agent_id (one_hot) + optional [speed, capacity, battery]
        input_dim = obs_dim + num_agents
        if self.heterogeneity_aware:
            input_dim += 3
            
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_head = nn.Linear(hidden_dim, act_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared_net(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value


class HeteroPolicyTrainer:
    def __init__(self, obs_dim, num_agents, act_dim, hidden_dim=32, learning_rate=1e-4, heterogeneity_aware=False):
        self.num_agents = num_agents
        self.heterogeneity_aware = heterogeneity_aware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = HeteroActorCriticNetwork(
            obs_dim, num_agents, act_dim, hidden_dim, heterogeneity_aware
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Buffers for episodic training
        self.log_probs = []
        self.values = []
        self.rewards = []

    def get_actions(self, obs_list, agents_props):
        """
        Takes a list of observations, one per agent, and returns the sampled actions.
        """
        actions = []
        step_log_probs = []
        step_values = []
        
        for i, obs in enumerate(obs_list):
            inputs = self._build_input(obs, i, agents_props[i]).unsqueeze(0).to(self.device)
            logits, value = self.network(inputs)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            actions.append(action.item())
            step_log_probs.append(dist.log_prob(action))
            step_values.append(value)
            
        self.log_probs.append(step_log_probs)
        self.values.append(step_values)
        
        return actions

    def store_rewards(self, rewards_list):
        """
        Store a step's rewards. rewards_list should optionally be castable to list of floats
        """
        self.rewards.append(rewards_list)
        
    def _build_input(self, obs, agent_id, agent_prop):
        one_hot = torch.zeros(self.num_agents)
        one_hot[agent_id] = 1.0
        
        obs_tensor = torch.FloatTensor(obs)
        
        if self.heterogeneity_aware:
            speed = agent_prop.get("speed", 1.0)
            # Support both original and mapped keys
            capacity = agent_prop.get("capacity", agent_prop.get("load_capacity", 10.0))
            battery = agent_prop.get("battery_capacity", agent_prop.get("battery", 100.0))
            
            cap_norm = capacity / 10.0
            bat_norm = battery / 100.0
            
            het_features = torch.tensor([speed, cap_norm, bat_norm], dtype=torch.float32)
            inputs = torch.cat([obs_tensor, one_hot, het_features])
        else:
            inputs = torch.cat([obs_tensor, one_hot])
            
        return inputs
        
    def train_episode(self, gamma=0.99):
        """
        Computes REINFORCE with baseline advantage and updates the network.
        Returns the mean actor and critic loss.
        """
        T = len(self.rewards)
        if T == 0:
            return 0.0, 0.0
            
        actor_loss = 0.0
        critic_loss = 0.0
        
        for agent_id in range(self.num_agents):
            # Compute discounted returns for this agent
            returns = []
            G = 0
            for t in reversed(range(T)):
                G = self.rewards[t][agent_id] + gamma * G
                returns.insert(0, G)
                
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            
            for t in range(T):
                log_prob = self.log_probs[t][agent_id].view(-1)[0]
                value = self.values[t][agent_id].view(-1)[0]
                ret = returns[t]
                
                # Advantage = Return - Value Baseline
                advantage = ret - value.detach()
                
                actor_loss = actor_loss - (log_prob * advantage)
                critic_loss = critic_loss + F.mse_loss(value, ret)
                
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        # Normalize losses for reporting
        total_steps = T * self.num_agents
        return (actor_loss.item() / total_steps), (critic_loss.item() / total_steps)
