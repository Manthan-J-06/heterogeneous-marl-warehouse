"""MAPPO whose per-robot extra inputs can change every step.

Same algorithm as algorithms/mappo.py (team reward, GAE, team-level clipped
ratio, one Adam), but the robot features ("props", shape (n_agents, props_dim))
are passed in at every step instead of being fixed at construction. Needed
for the varying-fleet experiment: the fleet changes each episode and the
current battery charge changes each step.

props_dim = 0  -> blind policy (obs + robot one-hot only).
props_dim > 0  -> aware policy (obs + robot one-hot + robot features; critic also
                  sees all robots' features).
"""
import numpy as np
import torch

from algorithms.mappo import MAPPOTrainer, Actor, CentralizedCritic


class DynamicMAPPOTrainer(MAPPOTrainer):
    def __init__(self, obs_dim, state_dim, n_agents, n_actions, cfg, device="cpu", props_dim=0):
        super().__init__(obs_dim, state_dim, n_agents, n_actions, cfg, device, agent_props=None)
        self.props_dim = props_dim
        self.dyn_aware = props_dim > 0
        if self.dyn_aware:
            self.actor = Actor(obs_dim, n_agents, n_actions, cfg["hidden_dim"], extra_dim=props_dim).to(device)
            self.critic = CentralizedCritic(state_dim + n_agents * props_dim, cfg["hidden_dim"]).to(device)
            self.optimizer = torch.optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg["learning_rate"])
        self._props_list = []

    def _t(self, x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

    def act(self, obs, state, props=None, record=False):
        obs_t = self._t(obs)
        state_t = self._t(state).unsqueeze(0)
        p = None
        if self.dyn_aware:
            p = self._t(props)
            state_t = torch.cat([state_t, p.reshape(1, -1)], dim=-1)
            if record:
                self._props_list.append(p.cpu().numpy())
        with torch.no_grad():
            dist = self.actor(obs_t, self.agent_onehot, p)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            value = self.critic(state_t).item()
        return actions.cpu().numpy().tolist(), logprobs.cpu().numpy(), value

    def update(self, last_state, last_props=None):
        T = len(self.buffer)
        team_rewards = [float(np.mean(r)) for r in self.buffer.rewards]

        with torch.no_grad():
            ls = self._t(last_state).unsqueeze(0)
            if self.dyn_aware:
                ls = torch.cat([ls, self._t(last_props).reshape(1, -1)], dim=-1)
            last_value = self.critic(ls).item()

        advantages, returns = self._compute_gae(team_rewards, self.buffer.values, self.buffer.dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_b = self._t(np.array(self.buffer.obs))
        state_b = self._t(np.array(self.buffer.states))
        act_b = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.long, device=self.device)
        old_logp_b = self._t(np.array(self.buffer.logprobs))
        adv_b = self._t(advantages)
        ret_b = self._t(returns)
        onehot_b = self.agent_onehot.unsqueeze(0).expand(T, -1, -1).reshape(T * self.n_agents, -1)

        props_b = None
        if self.dyn_aware:
            assert len(self._props_list) == T, "props must be recorded once per buffered step"
            pb = self._t(np.array(self._props_list))                    # (T, n, props_dim)
            props_b = pb.reshape(T * self.n_agents, -1)
            state_b = torch.cat([state_b, pb.reshape(T, -1)], dim=-1)

        tp, tv, te = 0.0, 0.0, 0.0
        for _ in range(self.epochs):
            dist = self.actor(obs_b.reshape(T * self.n_agents, -1), onehot_b, props_b)
            new_logp = dist.log_prob(act_b.reshape(-1)).reshape(T, self.n_agents)
            entropy = dist.entropy().reshape(T, self.n_agents).mean()
            ratio = torch.exp((new_logp - old_logp_b).mean(dim=1))
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((self.critic(state_b) - ret_b) ** 2).mean()
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 10.0)
            self.optimizer.step()
            tp += policy_loss.item(); tv += value_loss.item(); te += entropy.item()

        self.buffer.reset()
        self._props_list = []
        return {"policy_loss": tp / self.epochs, "value_loss": tv / self.epochs, "entropy": te / self.epochs}
