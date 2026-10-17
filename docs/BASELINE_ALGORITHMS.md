# Baseline Algorithm Reference: MAPPO and QMIX

Reference for the two baseline algorithms in this project, as implemented in this repository (`algorithms/mappo.py`, `algorithms/qmix.py`, `train_mappo.py`, `train_qmix.py` on `main`), plus the literature context for them. Section 1 explains the algorithms in general terms. Section 2 describes what our code actually does. Section 3 summarises what published benchmarks report on RWARE and related tasks.

**Sources (checked against the papers' text, 2026-10-17)**
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games" (MAPPO), arXiv 2103.01955 (NeurIPS 2022 Datasets and Benchmarks).
- Rashid et al., "Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (QMIX), ICML 2018 / JMLR 2020.
- Papoudakis et al., "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks", arXiv 2006.07869 (NeurIPS 2021 Datasets and Benchmarks). Introduces RWARE and EPyMARL.
- Papadopoulos et al., "An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks", arXiv 2502.04773 (AAMAS 2025).

Numbers quoted from the two benchmark papers were read from the papers' text and tables on the date above. Re-read the PDF tables before quoting any figure in the manuscript.

---

## 1. What each algorithm does

### MAPPO (Multi-Agent PPO)

MAPPO extends single-agent PPO to several agents using **Centralized Training with Decentralized Execution (CTDE)**:

- **Actors are decentralized.** Each agent picks an action from its own local observation, so no agent needs to see what the others see at execution time.
- **The critic is centralized during training.** One value function sees a global input (the global state, or the concatenation of all agents' observations). It is used to compute training targets and discarded at execution time.
- **It is on-policy.** It collects a batch of fresh experience with the current policy, runs a few gradient epochs on it, then discards it.
- **The update is PPO's clipped surrogate objective**, which limits how far the policy can move per update.
- **Parameter sharing** (one set of weights for all agents, usually with an agent ID in the input so behaviour can still differ) is common practice.

The centralized critic exists to fight non-stationarity: from one agent's point of view the environment keeps changing because the others are learning too.

### QMIX

QMIX is a **value-based, off-policy** method built on **value decomposition**:

- Each agent has its own action-value function Q_i over its local observation and action. Execution is decentralized.
- During training a **mixing network** combines the Q_i values into a joint value Q_tot, conditioned on the global state.
- The mixing weights are constrained to be non-negative (produced by hypernetworks from the state). This **monotonicity constraint** means that if every agent greedily maximises its own Q_i, the joint action is also greedy for Q_tot.
- It is off-policy: experience goes into a replay buffer and is reused across many updates.
- The original QMIX uses recurrent (GRU) agent networks. Ours does not (Section 2).

### Structural differences

| | MAPPO | QMIX |
|---|---|---|
| Learning paradigm | Policy gradient (actor-critic), on-policy | Value-based (Q-learning), off-policy |
| What is centralized in training | The critic | The Q-value mixing function |
| How cooperation is enforced | Shared critic gives a consistent training signal | Monotonicity constraint on the mixer |
| Data usage | Fresh batch each update, then discarded | Replay buffer, reused |
| Output at execution | A stochastic policy per agent | A Q-function per agent (greedy or epsilon-greedy) |
| Reward | Can use shared or individual rewards in general | Needs a single team reward |

---

## 2. How our implementation works (verified from the code on `main`)

Both algorithms are our own PyTorch implementations, not EPyMARL. Hyperparameter names therefore follow our YAML keys, not EPyMARL's.

### Shared design choices

- **Parameter sharing** across agents. Agent identity is passed as a one-hot vector.
- **MLP networks, no recurrence.** Two hidden layers of `hidden_dim` units (64).
- **Team reward = mean of the per-agent rewards.** Both trainers average the per-agent reward vector. A per-agent penalty such as `energy_weight` is therefore averaged into the team reward, and individual credit is lost.
- **Centralized input** is the concatenation of all agents' observations (`state_dim = obs_dim * n_agents`).
- **Gradient clipping** at norm 10.0, Adam optimizer.

### MAPPO (`algorithms/mappo.py`, `train_mappo.py`)

| Component | Implementation |
|---|---|
| Actor | MLP, Tanh. Input: observation + one-hot agent ID (+ properties vector in the aware variant). Categorical output over 5 actions |
| Critic | MLP, Tanh. Input: global state (+ flattened properties of all agents in the aware variant). Scalar output |
| Advantage | GAE (`gamma`, `gae_lambda`) on the team reward, normalised to zero mean and unit variance per update |
| Policy objective | Clipped surrogate with a **team-level ratio**: exp of the mean over agents of (new log-prob − old log-prob). This differs from per-agent ratios in the original MAPPO formulation |
| Loss | policy loss + `value_loss_coef` x value MSE − `entropy_coef` x entropy |
| Optimizer | One Adam over actor and critic parameters, one learning rate (no separate critic learning rate) |
| Epochs and batching | `ppo_epochs` full-batch passes over each rollout. No minibatches |
| Value normalization | None |
| Heterogeneity-aware variant (`--aware`) | Appends `[speed, capacity/10, battery/1000]` per agent to the actor input, and the flattened properties of all agents to the critic input. The blind variant gets neither |

Hyperparameters in `configs/mappo_rware_homogeneous.yaml`: gamma 0.99, gae_lambda 0.95, clip_eps 0.2, entropy_coef 0.01, value_loss_coef 0.5, ppo_epochs 4, hidden_dim 64, learning_rate 3e-4, rollout_len 400, total_env_steps 500,000.

The Phase 4b fleet runs read the same keys from `configs/low_variance_fleet.yaml`: gamma 0.99, gae_lambda 0.95, clip_eps 0.2, entropy_coef 0.01, value_loss_coef 0.5, ppo_epochs 4, hidden_dim 64, rollout_len 400, and **learning_rate 1e-4** (one key shared with QMIX; the homogeneous MAPPO config uses 3e-4). They ran for 30,000 steps, which is 75 updates of 400 steps, or about 300 gradient steps (75 updates x 4 epochs). The config default is 500,000 steps (1,250 updates, about 5,000 gradient steps). That config also uses the temporary easy environment (`rware-tiny-4ag-easy-v2`, request queue 8) and batteries loosened 10x (850 to 1000).

### QMIX (`algorithms/qmix.py`, `train_qmix.py`)

| Component | Implementation |
|---|---|
| Agent network | Shared MLP, ReLU. Input: observation + one-hot agent ID. Output: Q per action |
| Mixer | Hypernetwork mixer with absolute-value weights, mixing hidden size 32, hypernetwork hidden size 64, ELU |
| Replay | Transition-level buffer (valid because the agent network is memoryless), capacity 50,000 |
| Update frequency | One gradient step per environment step once the buffer holds a batch |
| Targets | Double-Q: online network selects the next action, target network evaluates it, target mixer mixes |
| Loss | Huber (smooth L1) |
| Optimizer | Adam, learning rate 1e-4, grad-norm clip 10 |
| Target network | Hard copy every `target_update_interval` = 400 training steps |
| Exploration | Epsilon-greedy, linear 1.0 to 0.05 over 100,000 steps |
| Batch size | 64 |
| Heterogeneity-aware variant | **None.** QMIX on `main` is heterogeneity-blind only |

Hyperparameters in `configs/qmix_rware_homogeneous.yaml`: gamma 0.99, hidden_dim 64, mixing_hidden_dim 32, learning_rate 1e-4, buffer_size 50,000, batch_size 64, target_update_interval 400, epsilon 1.0 to 0.05 over 100,000 steps, total_env_steps 500,000.

**Stability changes.** The first QMIX version diverged (loss in the billions). The current version uses Huber loss, Double-Q targets, a lower learning rate (5e-4 to 1e-4) and a slower target update (200 to 400). The code comments report that Huber loss alone cut the loss by about 600x in ablation testing.

**Short-run caveat.** The epsilon schedule decays over 100,000 steps. A 20,000-step run only reaches epsilon of about 0.81 and a 30,000-step run about 0.72, so short QMIX runs are still mostly random.

---

## 3. Literature: how do MAPPO and QMIX compare on RWARE and related tasks?

### RWARE (most relevant to this project)

- **Papoudakis et al. 2021** (arXiv 2006.07869). On the RWARE tasks MAPPO achieves the best returns of all evaluated algorithms (Table 3, Section 5.2), though not always significantly. VDN and QMIX show no learning, and the paper attributes this mainly to RWARE's sparse reward: value decomposition needs sufficiently dense rewards, and the agents' utilities are rarely above zero.
- **Papadopoulos et al. 2025** (arXiv 2502.04773, AAMAS 2025), on the "-hard" RWARE tasks:
  - QMIX "shows mediocre performance across most tasks, with complete failure in some (e.g., RWARE and many LBF tasks)" (Section 5.2.2).
  - QPLEX "notably improves over QMIX in RWARE" (Section 5.2.2). In Table 5 on tiny-4ag-hard QMIX scores 0.00 ± 0.00, QPLEX 11.73 ± 10.81 and MAPPO 17.1 ± 5.31.
  - MAPPO "significantly outperforms MAA2C in RWARE" (Section 5.2.2), but in the RWARE tasks where MAPPO was best in earlier work, HAPPO, MAT-DEC and CDS outperform it.
  - **Caution:** Section 5.2.1 says "QPLEX, or even QMIX" converges on tiny-4ag-hard, but Table 5 gives QMIX 0.00 there. Cite the table, not that sentence, for QMIX.

### Other benchmarks

- **Yu et al. 2022** (MAPPO paper, arXiv 2103.01955): "MAPPO and IPPO perform at least as well as QMix in most maps despite using the same number of samples" on SMAC (Section 4.3). The authors use 15 PPO epochs on easy tasks and 10 or 5 on difficult ones, and report 15 epochs led MAPPO to a suboptimal policy on a hard map (Section 5.3). They also found value normalization "never hurts training and often improves the final performance" (Section 5.1). Our MAPPO has no value normalization.

### Papers from our literature review spreadsheet most relevant to this project

(From `MARL_Literature_Review.xlsx`, as summarised there by the team; not independently re-checked.)
- Choi et al. 2022, multi-AGV control in warehouses: plain QMIX alone was insufficient to raise productivity; it assumes homogeneous AGVs.
- Krnjaic et al. 2023, scalable MARL for warehouse logistics with robotic and human co-workers: hierarchical MARL with heterogeneous agents and individual rewards; the closest prior work on heterogeneity in warehouse MARL.
- Wibisono et al. 2025, energy-aware MARL for multi-AUV data collection: a survey and case study on energy efficiency, not a warehouse setting.

### Implications for this project

- The literature favours MAPPO over QMIX on RWARE, and attributes QMIX's failure to sparse reward.
- Our environment is `rware-tiny-4ag-easy-v2`, not the "-hard" variants those papers used. Check how the variants differ before comparing numbers.
- Our Phase 4b runs used 30,000 steps, while the configs default to 500,000. Check the benchmark papers' training budgets before comparing.
- In our runs the blind MAPPO policy was still close to uniform after 30,000 steps (entropy about 1.58 against a maximum of ln 5 = 1.61). With only about 300 gradient steps at a learning rate of 1e-4, this may be under-training as much as sparse reward. We have not yet run the full 500,000-step budget, so no conclusion about learning can be drawn from the Phase 4b runs.

### Claims removed from the earlier draft
- A paper attributed to "Christianos et al., arXiv 2302.03439": that identifier is Schäfer et al., "Ensemble Value Functions for Efficient Exploration in MARL", a value-based method, so it does not support the RWARE claim made for it.
- SMACv2 results and a "2025 summary" about MAA2C and warehouse tasks: not verified.
- "15 to 5 epochs" attributed to an unnamed benchmarking report: the source is Yu et al. 2022 (Section 5.3).

---

## 4. Hyperparameters that matter in our implementation

**MAPPO:** `learning_rate`, `ppo_epochs`, `entropy_coef`, `clip_eps`, `rollout_len`. The literature suggests value normalization may help, and ours has none.

**QMIX:** the epsilon schedule (`epsilon_decay_steps`), `learning_rate`, `target_update_interval`, `buffer_size` and `batch_size`.

**Seeds:** our own Phase 4b runs show training-seed variance as large as the effects of interest (see `PHASE5_HANDOFF.md`), so any comparison needs several training seeds per condition.

---

## 5. Open items

- Run the full 500,000-step budget (and the homogeneous config's learning rate of 3e-4) for aware and blind MAPPO before concluding anything about learning.
- Decide whether to build a heterogeneity-aware QMIX variant, or restrict the aware-vs-blind comparison to MAPPO.
- Decide whether to keep the team-level ratio in MAPPO or move to per-agent ratios, and note the choice in the paper.
- Decide how `energy_weight` should be handled, given that both trainers average per-agent rewards.
