"""
MAPPO for a homogeneous fleet — parameter-shared actor (decentralized
execution) + centralized critic conditioned on global state (Yu et al.,
2022, "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
Games").

Design choices stated explicitly:
  - Shared actor across agents (standard for homogeneous fleets), agent
    identity passed as one-hot, same rationale as the QMIX file.
  - Centralized critic takes the concatenated global state, matching the
    CTDE pattern used for the QMIX mixing network so the two baselines
    are as comparable as possible.
  - On-policy rollout buffer of fixed length (`rollout_len` in config),
    standard synchronous PPO-style collection.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs, agent_onehot):
        logits = self.net(torch.cat([obs, agent_onehot], dim=-1))
        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs, self.states, self.actions = [], [], []
        self.logprobs, self.rewards, self.dones, self.values = [], [], [], []

    def add(self, obs, state, actions, logprobs, reward, done, value):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self):
        return len(self.rewards)


class MAPPOTrainer:
    def __init__(self, obs_dim, state_dim, n_agents, n_actions, cfg, device="cpu"):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_eps = cfg["clip_eps"]
        self.entropy_coef = cfg["entropy_coef"]
        self.value_loss_coef = cfg["value_loss_coef"]
        self.epochs = cfg["ppo_epochs"]
        self.device = device

        self.actor = Actor(obs_dim, n_agents, n_actions, cfg["hidden_dim"]).to(device)
        self.critic = CentralizedCritic(state_dim, cfg["hidden_dim"]).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg["learning_rate"],
        )
        self.agent_onehot = torch.eye(n_agents, device=device)
        self.buffer = RolloutBuffer()

    def act(self, obs, state):
        """Returns actions (list[int]), logprobs (np.array[n_agents]), value (float)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs_t, self.agent_onehot)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            value = self.critic(state_t).item()
        return actions.cpu().numpy().tolist(), logprobs.cpu().numpy(), value

    def _compute_gae(self, rewards, values, dones, last_value):
        # rewards/values/dones: (T,) team-level (mean over agents already applied by caller)
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values_ext = values + [last_value]
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, last_state):
        T = len(self.buffer)
        team_rewards = [float(np.mean(r)) for r in self.buffer.rewards]

        with torch.no_grad():
            last_value = self.critic(
                torch.as_tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            ).item()

        advantages, returns = self._compute_gae(
            team_rewards, self.buffer.values, self.buffer.dones, last_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_b = torch.as_tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)  # (T, n_agents, obs_dim)
        state_b = torch.as_tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)  # (T, state_dim)
        act_b = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.long, device=self.device)  # (T, n_agents)
        old_logp_b = torch.as_tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=self.device)  # (T, n_agents)
        adv_b = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)  # (T,)
        ret_b = torch.as_tensor(returns, dtype=torch.float32, device=self.device)  # (T,)

        agent_onehot_batch = self.agent_onehot.unsqueeze(0).expand(T, -1, -1)

        total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
        for _ in range(self.epochs):
            dist = self.actor(obs_b.view(T * self.n_agents, -1), agent_onehot_batch.reshape(T * self.n_agents, -1))
            new_logp = dist.log_prob(act_b.view(-1)).view(T, self.n_agents)
            entropy = dist.entropy().view(T, self.n_agents).mean()

            ratio = torch.exp((new_logp - old_logp_b).mean(dim=1))  # team-level ratio (mean over agents)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.critic(state_b)
            value_loss = ((values - ret_b) ** 2).mean()

            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 10.0
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        self.buffer.reset()
        return {
            "policy_loss": total_policy_loss / self.epochs,
            "value_loss": total_value_loss / self.epochs,
            "entropy": total_entropy / self.epochs,
        }

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
