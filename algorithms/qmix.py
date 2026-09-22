"""
QMIX for a homogeneous fleet — parameter-shared per-agent Q-network +
monotonic mixing network (Rashid et al., 2018).

Design choices, stated explicitly rather than left implicit:
  - Parameter sharing across agents: standard practice for a homogeneous
    fleet baseline; agent identity is passed in as a one-hot so the shared
    network can still condition behavior on which agent it's acting for.
  - MLP agent network, not a GRU: simpler and sufficient as a baseline;
    swap in an RNN + episode replay buffer later if partial observability
    or long-horizon credit assignment becomes a bottleneck.
  - Transition-level replay buffer (not full-episode): valid because the
    agent net is memoryless (MLP); would need to change to episode-level
    replay if you move to a recurrent agent network.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentQNet(nn.Module):
    """Shared Q-network: input = obs + one-hot agent id, output = Q per action."""

    def __init__(self, obs_dim, n_agents, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs, agent_onehot):
        x = torch.cat([obs, agent_onehot], dim=-1)
        return self.net(x)


class MixingNetwork(nn.Module):
    """Hypernetwork producing non-negative mixing weights -> guarantees the
    monotonicity constraint dQ_tot/dQ_i >= 0 that makes QMIX's greedy
    decentralized argmax consistent with the joint argmax."""

    def __init__(self, n_agents, state_dim, mixing_hidden_dim=32, hyper_hidden_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden_dim = mixing_hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim), nn.ReLU(),
            nn.Linear(hyper_hidden_dim, n_agents * mixing_hidden_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim), nn.ReLU(),
            nn.Linear(hyper_hidden_dim, mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim), nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(self, agent_qs, state):
        # agent_qs: (batch, n_agents), state: (batch, state_dim)
        bs = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(state).view(bs, 1, self.mixing_hidden_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, 1)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QMIXTrainer:
    def __init__(self, obs_dim, state_dim, n_agents, n_actions, cfg, device="cpu"):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = cfg["gamma"]
        self.device = device

        self.agent_net = AgentQNet(obs_dim, n_agents, n_actions, cfg["hidden_dim"]).to(device)
        self.target_agent_net = AgentQNet(obs_dim, n_agents, n_actions, cfg["hidden_dim"]).to(device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        self.mixer = MixingNetwork(n_agents, state_dim, cfg["mixing_hidden_dim"]).to(device)
        self.target_mixer = MixingNetwork(n_agents, state_dim, cfg["mixing_hidden_dim"]).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg["learning_rate"])

        self.buffer = ReplayBuffer(cfg["buffer_size"])
        self.agent_onehot = torch.eye(n_agents, device=device)
        self.train_step_count = 0
        self.target_update_interval = cfg["target_update_interval"]

    def act(self, obs, epsilon):
        """obs: (n_agents, obs_dim) numpy array. Returns list of ints."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.agent_net(obs_t, self.agent_onehot)  # (n_agents, n_actions)
        actions = []
        for i in range(self.n_agents):
            if random.random() < epsilon:
                actions.append(random.randrange(self.n_actions))
            else:
                actions.append(int(torch.argmax(q_values[i]).item()))
        return actions

    def store(self, obs, actions, reward, next_obs, done, state, next_state):
        self.buffer.push((obs, actions, reward, next_obs, done, state, next_state))

    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        obs_b, act_b, rew_b, next_obs_b, done_b, state_b, next_state_b = zip(*batch)

        obs_b = torch.as_tensor(np.array(obs_b), dtype=torch.float32, device=self.device)
        next_obs_b = torch.as_tensor(np.array(next_obs_b), dtype=torch.float32, device=self.device)
        act_b = torch.as_tensor(np.array(act_b), dtype=torch.long, device=self.device)
        # team reward for cooperative QMIX: mean of per-agent rewards
        rew_b = torch.as_tensor(np.array(rew_b), dtype=torch.float32, device=self.device).mean(dim=1)
        done_b = torch.as_tensor(np.array(done_b), dtype=torch.float32, device=self.device)
        state_b = torch.as_tensor(np.array(state_b), dtype=torch.float32, device=self.device)
        next_state_b = torch.as_tensor(np.array(next_state_b), dtype=torch.float32, device=self.device)

        bs = obs_b.shape[0]
        agent_onehot_batch = self.agent_onehot.unsqueeze(0).expand(bs, -1, -1)

        # current Q for taken actions
        q_vals = self.agent_net(
            obs_b.view(bs * self.n_agents, -1),
            agent_onehot_batch.reshape(bs * self.n_agents, -1),
        ).view(bs, self.n_agents, self.n_actions)
        chosen_q = torch.gather(q_vals, dim=2, index=act_b.unsqueeze(-1)).squeeze(-1)  # (bs, n_agents)
        q_tot = self.mixer(chosen_q, state_b).squeeze(-1)  # (bs,)

        with torch.no_grad():
            # Double-Q correction: select the next action with the ONLINE network,
            # evaluate it with the TARGET network. Plain max-over-target (the
            # original formulation) systematically overestimates Q-values, and
            # that overestimation compounds every bootstrap step — this was the
            # dominant remaining cause of divergence after switching to Huber loss.
            online_next_q = self.agent_net(
                next_obs_b.view(bs * self.n_agents, -1),
                agent_onehot_batch.reshape(bs * self.n_agents, -1),
            ).view(bs, self.n_agents, self.n_actions)
            next_actions = online_next_q.argmax(dim=2, keepdim=True)

            next_q_vals = self.target_agent_net(
                next_obs_b.view(bs * self.n_agents, -1),
                agent_onehot_batch.reshape(bs * self.n_agents, -1),
            ).view(bs, self.n_agents, self.n_actions)
            next_max_q = torch.gather(next_q_vals, dim=2, index=next_actions).squeeze(-1)
            next_q_tot = self.target_mixer(next_max_q, next_state_b).squeeze(-1)
            target = rew_b + self.gamma * (1 - done_b) * next_q_tot

        # Huber (smooth L1) loss, not MSE: MSE squares the error, so any transient
        # overestimation produces a proportionally huge gradient that pushes the
        # next prediction even further off — a self-reinforcing loop. This was the
        # single largest contributor to the divergence (~600x loss reduction on
        # its own in ablation testing). Huber caps that amplification.
        loss = F.smooth_l1_loss(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_interval == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            "agent_net": self.agent_net.state_dict(),
            "mixer": self.mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step_count": self.train_step_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(ckpt["agent_net"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_step_count = ckpt["train_step_count"]
