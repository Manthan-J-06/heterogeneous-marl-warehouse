# Baseline Algorithm Reference: MAPPO and QMIX

Internal reference on the two baseline algorithms for this project, and the EPyMARL framework we'll likely run them through. Written for teammates who haven't worked with either before.

**Sources**
- MAPPO paper: "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games" (Yu et al., arXiv 2103.01955 / NeurIPS 2022)
- QMIX paper: "Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (Rashid et al., JMLR 2020 / ICML 2018)
- EPyMARL: https://github.com/uoe-agents/epymarl (framework README, `src/config` structure)
- PyMARL (EPyMARL's base) default config: https://github.com/oxwhirl/pymarl/blob/master/src/config/default.yaml
- Benchmark paper (introduces RWARE, evaluates MAPPO/QMIX on it): "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks" (Papoudakis et al., NeurIPS Datasets & Benchmarks 2021)
- Follow-up: "An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks" (AAMAS 2025, arXiv 2502.04773)
- SMACv2 paper (Ellis et al., arXiv 2212.07489), for an additional QMIX vs MAPPO data point outside RWARE

> **Verification status.** Section 2's exact default hyperparameter values come from the public PyMARL `default.yaml` (the base EPyMARL is built on) plus a general survey of PPO-style configs, not from EPyMARL's own `src/config/algs/mappo.yaml` and `qmix.yaml`, which I could not open directly. Values from PyMARL's own file are marked confirmed; the rest are marked **[VERIFY]** and should be checked against the actual files once the repo is cloned locally (path given in Section 2). Section 3's comparison did not use `MARL_Literature_Review.xlsx` because I don't have access to it — see the note at the top of that section.
>
> **Note added when adding this doc to the repo:** this project uses custom `algorithms/qmix.py` and `algorithms/mappo.py` implementations, not EPyMARL directly — so Section 2's exact hyperparameter *names* won't map 1:1 to our actual configs (`configs/qmix_rware_homogeneous.yaml`, `configs/mappo_rware_homogeneous.yaml`). The algorithm explanations in Section 1 and the literature comparison in Section 3 still apply regardless of framework. Section 3's cross-check against `MARL_Literature_Review.xlsx` is still an open action item.

---

## 1. What each algorithm does

### MAPPO (Multi-Agent PPO)

MAPPO extends single-agent PPO (Proximal Policy Optimization) to multi-agent settings using **Centralized Training with Decentralized Execution (CTDE)**:

- **Actors are decentralized.** Each agent has its own policy that picks an action using only its own local observation. At deployment/execution time, no agent needs to see what any other agent sees.
- **The critic is centralized during training.** A single value function is trained on the global state (or the concatenation of all agents' observations, if no true global state is available). This critic is only used to compute training targets; it is thrown away at execution time.
- **It's on-policy.** Like PPO, it collects a batch of fresh experience with the current policy, does a handful of gradient update epochs on that batch, then throws the batch away. Training data is never reused across many updates the way replay-buffer methods do.
- **The core update is PPO's clipped surrogate objective** applied per-agent, using an advantage computed from the shared centralized critic. This clipping is what keeps policy updates from moving too far in one step, which stabilizes training.
- **Parameter sharing** (one set of weights used by every agent, usually with an agent ID appended to the input so agents can still behave differently) is common practice and is what EPyMARL does by default, though it can be turned off.

Key intuition: the centralized critic exists to fight *non-stationarity* — from any one agent's point of view, the environment keeps changing because the other agents are also learning. Seeing the global picture during training gives the critic a more stable target to learn against, even though each actor still has to act on partial information.

### QMIX

QMIX is a **value-based**, off-policy method built around **value decomposition**:

- Every agent has its own action-value function, Q_i, over its own local observation and action. These are the only things used at execution time, so execution is decentralized here too.
- During training, a **mixing network** combines all the individual Q_i values into a single joint value, Q_tot, conditioned on the global state.
- The mixing network's weights are constrained to be non-negative (produced by "hypernetworks" that take the global state as input). This enforces a **monotonicity constraint**: increasing any individual agent's Q_i can never decrease Q_tot. That constraint is the core trick — it guarantees that if every agent greedily maximizes its own Q_i, the result is also the greedy (argmax) action for the joint Q_tot. This is what lets a centrally-trained value function be executed in a fully decentralized way.
- It's **off-policy**: experience is stored in a replay buffer and reused across many gradient updates, unlike MAPPO's throw-away-the-batch style. This can make it more sample-efficient in principle, but training is generally less stable and more sensitive to hyperparameters than PPO-style methods.
- Individual agent Q-networks are typically recurrent (an RNN/GRU) to help with partial observability, following the same convention as DRQN.

### The core structural difference

| | MAPPO | QMIX |
|---|---|---|
| Learning paradigm | Policy gradient (actor-critic), on-policy | Value-based (Q-learning), off-policy |
| What's centralized in training | The critic (a value estimate) | The Q-value mixing function |
| How cooperation is enforced | Shared critic gives a consistent training signal | Monotonicity constraint on the mixing network |
| Data usage | Fresh batch each update, then discarded | Replay buffer, reused many times |
| Output at execution | A stochastic policy per agent | A Q-function per agent (act greedily / epsilon-greedy) |
| Typically better suited to | General-sum or shared reward, any reward density | Common (shared) reward, historically assumed dense reward |

Both are CTDE methods, and both scale reasonably with agent count, but they get there in different ways: MAPPO by regularizing a policy-gradient update with a global critic, QMIX by constraining how per-agent Q-values are allowed to combine.

---

## 2. How EPyMARL implements both, and key hyperparameters

EPyMARL (an extension of PyMARL) is the framework this benchmarking paper — and most of the RWARE literature — uses, and it directly supports RWARE via `env-config=gymma` with `env_args.key="rware:rware-tiny-2ag-v2"` style environment IDs.

**Config file layout** (from the EPyMARL repo, once cloned):
- `src/config/default.yaml` — shared defaults (experiment length, logging, general RL settings)
- `src/config/algs/mappo.yaml` and `src/config/algs/qmix.yaml` — per-algorithm hyperparameters, which override the defaults
- `src/config/envs/` — per-environment settings (e.g. `gymma.yaml` for the wrapper RWARE uses)

Run pattern: `python src/main.py --config=mappo --env-config=gymma with env_args.key="rware:rware-tiny-2ag-v2" ...`

**One important framework-level constraint**, confirmed from the EPyMARL README: QMIX only supports environments with a single common (shared) reward. MAPPO can be run either with a common reward or with individual per-agent rewards (`common_reward=False`). If our environment ever moves toward heterogeneous, individually-rewarded agents, this is a hard constraint on QMIX, not just a tuning knob.

### Hyperparameters confirmed from PyMARL's `default.yaml` (shared base, both algorithms build on this)

| Hyperparameter | Default | What it controls |
|---|---|---|
| `gamma` | 0.99 | Discount factor — how much future reward is worth relative to immediate reward. |
| `batch_size` | 32 | Number of episodes used per training update. |
| `buffer_size` | 32 | Size of the replay buffer (episodes). QMIX-relevant; MAPPO is on-policy and doesn't reuse old data the same way. |
| `lr` | 0.0005 | Learning rate for the agent network(s). |
| `critic_lr` | 0.0005 | Learning rate for the critic (relevant to actor-critic methods like MAPPO). |
| `optim_alpha` | 0.99 | RMSProp optimizer smoothing constant. |
| `optim_eps` | 0.00001 | RMSProp epsilon (numerical stability term). |
| `grad_norm_clip` | 10 | Caps the L2 norm of gradients — prevents destructively large updates. |
| `agent` | `"rnn"` | Default agent network is recurrent, to handle partial observability. |
| `rnn_hidden_dim` | 64 | Size of the hidden state in the recurrent agent network. |
| `obs_agent_id` | True | Whether each agent's one-hot ID is appended to its observation (needed to tell agents apart under parameter sharing). |
| `obs_last_action` | True | Whether the agent's previous action is appended to its observation. |

### Hyperparameters that are QMIX-specific **[VERIFY exact defaults in qmix.yaml]**

- **Exploration schedule** (`epsilon_start`, `epsilon_finish`, `epsilon_anneal_time`, or similarly named): QMIX acts epsilon-greedily during training, annealing from mostly-random to mostly-greedy over some number of timesteps. This is one of the most sensitive settings for QMIX and is a common target for tuning.
- **Target network update rate** (`target_update_interval`, sometimes called `tau` for soft updates): how often/how much the target Q-network is refreshed from the online network. Affects training stability.
- **Mixing network hidden dimension**: size of the mixing network's hypernetwork layers.
- **Double-Q usage**: whether QMIX uses double Q-learning style target computation (a common stability improvement over vanilla QMIX).

### Hyperparameters that are MAPPO-specific **[VERIFY exact defaults in mappo.yaml]**

Based on standard MAPPO/PPO implementations (including EPyMARL's stated design goal of "consistency of implementation" across its actor-critic algorithms), expect config keys similar to:
- **`eps_clip`** (~0.2): the PPO clipping parameter — how far a new policy is allowed to move from the old one in a single update, in probability-ratio terms.
- **Number of epochs per update** (often called `epochs` or similar): how many passes of gradient descent are done over one collected batch before it's discarded. The MAPPO paper found this is very sensitive on hard tasks — dropping from 15 to 5 epochs meaningfully changed results.
- **`entropy_coef`**: weight on an entropy bonus that encourages exploration by discouraging the policy from becoming overconfident too early.
- **`value_loss_coef`** (often called `critic_coef`): weight on the critic's loss term relative to the policy loss term in the combined objective.
- **GAE lambda** (`gae_lambda` or `lambda`): controls the bias/variance trade-off in advantage estimation.
- **Value normalization / standardised returns**: EPyMARL's README-adjacent documentation notes it supports return/reward standardization as an implementation detail; the MAPPO paper found normalizing value targets consistently helps and rarely hurts.
- **Minibatch size**: batch is often split into minibatches for the multiple gradient epochs.

### What generally controls what (cheat sheet for later tuning)

| Category | MAPPO knobs | QMIX knobs |
|---|---|---|
| How fast it learns | `lr`, `critic_lr` | `lr` |
| How much it discounts the future | `gamma` (shared) | `gamma` (shared) |
| How much data per update | `batch_size`, minibatch size, epochs | `batch_size`, `buffer_size` |
| Exploration | Entropy bonus (`entropy_coef`) | Epsilon-greedy schedule |
| Update stability | Clip range (`eps_clip`), grad norm clip, value normalization | Target network update rate, grad norm clip, double-Q |
| Network capacity | `rnn_hidden_dim`, actor/critic hidden layers | `rnn_hidden_dim`, mixing network hidden dim |

---

## 3. Literature comparison: when does MAPPO tend to differ from QMIX?

> **Note on sources.** I could not access `MARL_Literature_Review.xlsx` while writing this (it lives in the repo, not in my available context). The comparison below draws on the two benchmark papers that specifically evaluate RWARE (most relevant to us) plus one additional paper (SMACv2) for a data point in a different environment. **Someone should cross-check this section against the actual papers listed in the spreadsheet** and fold in anything it says that isn't covered here — that's flagged again at the end of this section.

**On RWARE specifically (most relevant to this project).** This is the strongest and most consistent finding across sources:
- The original EPyMARL benchmark paper found that in most RWARE tasks, MAPPO (and MAA2C) clearly outperformed the CTDE value-based methods (QMIX, VDN, COMA), and that training state-action value functions appears challenging in RWARE tasks with sparse rewards, leading to very low performance of the remaining CTDE algorithms (COMA, VDN and QMIX). In fact, VDN and QMIX do not exhibit any learning in RWARE, similar to IQL, COMA and MADDPG. MAPPO's on-policy surrogate objective was specifically credited with giving it strong sample efficiency in RWARE, and its achieved returns exceed the returns of all other algorithms in RWARE tasks, though not always by a significant margin.
- A 2025 extended benchmarking paper revisited this. It found that value-decomposition methods can sometimes converge in sparse-reward settings after all — including on the specific `RWARE tiny-4ag-hard` configuration — contrasting with the earlier conclusion that value decomposition methods require sufficiently dense rewards to learn. But its general summary of QMIX in RWARE was still that QMIX shows mediocre performance across most tasks, with complete failure in some (e.g., RWARE and many LBF tasks), while a related value-decomposition variant, QPLEX, notably improves over QMIX in RWARE.
- A separate exploration-focused paper reported a similar pattern in RWARE: IPPO and MAPPO perform well in RWARE with MAPPO reaching the highest evaluation returns in two smaller RWARE tasks, while QMIX-family methods failed to learn at all in that setting, consistent with independent value-based learners generally outperforming centralized value-decomposition in this specific environment.

**Takeaway for RWARE-like environments:** the literature is fairly consistent that plain QMIX struggles here because of reward sparsity combined with the difficulty of learning a good monotonic value decomposition, whereas MAPPO's on-policy, clipped-update approach tends to learn something useful faster. This is directly relevant to our warehouse setting.

**On other benchmarks (context, not RWARE).** The picture is less one-sided elsewhere:
- On SMAC (StarCraft micromanagement, competitive-flavored cooperative tasks), one paper found MAPPO achieves the highest rewards with the fewest amount of samples on all tasks when given a comparable training budget, and that it is an order of magnitude faster than off-policy baselines since it takes fewer gradient updates.
- The original QMIX-vs-MAPPO comparisons in the SMAC literature were more mixed: an early benchmarking effort found that in easy and medium environments, MAPPO performs as well as, if not better than, QMix, but in the hardest maps, MAPPO underperforms relative to QMix — though this gap was also sensitive to how many PPO epochs per update were used.
- On the newer SMACv2 benchmark, results leaned toward QMIX: QMIX generally performs better than MAPPO across most scenarios, strongly outperforming it in two Protoss scenarios, and is more sample efficient where the two were roughly comparable elsewhere.
- A broader 2025 fully-cooperative benchmark summarized the overall pattern as: MAA2C particularly excels in complex sparse-reward scenarios, while MAPPO shows strength in warehouse management and navigation tasks, with QMIX and QPLEX described as more consistent standard baselines overall but weaker specifically in the sparsest, most cooperative settings.

**Broad pattern to take away:**

| Setting | Tends to favor |
|---|---|
| Sparse/delayed reward, purely cooperative tasks (RWARE is the clearest example) | MAPPO |
| Warehouse / navigation-style tasks generally | MAPPO |
| Competitive-flavored "team vs team" tasks with denser reward signals (some SMAC maps) | Can go either way; QMIX competitive or ahead when well-tuned |
| Environments needing individual (non-shared) rewards | MAPPO only — QMIX isn't built for this |
| Very large agent counts | Value-decomposition methods (QMIX/QPLEX) can degrade; actor-critic methods tend to hold up better in some studies |

Given that our project is a warehouse coordination task closely related to RWARE, **the literature suggests MAPPO is the stronger default baseline to get running first**, with QMIX kept as a comparison point that may need either careful tuning or an improved variant (e.g. QPLEX) to be competitive here.

**Action item:** please cross-check `MARL_Literature_Review.xlsx` against this section and note any paper-specific findings (e.g. any RWARE-adjacent or heterogeneous-agent papers your team specifically flagged) that should be added.

*Findings in this section are drawn from: Papoudakis et al. 2021 (NeurIPS Datasets & Benchmarks) for the RWARE/EPyMARL results; the AAMAS 2025 extended benchmarking paper (arXiv 2502.04773) for the QPLEX/QMIX follow-up; Christianos et al. (arXiv 2302.03439) for the exploration-focused RWARE result; Yu et al. 2022 (MAPPO paper) and an early QMIX-vs-MAPPO SMAC benchmarking report for the SMAC comparisons; and Ellis et al. (SMACv2, arXiv 2212.07489) for the SMACv2 data point.*

---

## 4. Hyperparameters most commonly tuned in the literature

Across the papers reviewed above (and general MARL-tuning practice), the hyperparameters that repeatedly show up as the main levers people search over are:

**For MAPPO / on-policy actor-critic methods:**
1. **Learning rate** — for both actor and critic; frequently swept on a log scale.
2. **Number of PPO epochs per update** — explicitly called out in the QMix-vs-MAPPO benchmarking paper as changing hard-map results significantly (15 → 5 epochs).
3. **Entropy coefficient** — trades off exploration vs. exploitation; environment-dependent.
4. **Clip range (`eps_clip`)** — controls how aggressive policy updates can be.
5. **Value normalization / return standardization** — found to help consistently and rarely hurt in the PPO cooperative-games paper.
6. **Batch/rollout length and minibatch size** — affects gradient variance and wall-clock efficiency.

**For QMIX / value-decomposition methods:**
1. **Exploration schedule (epsilon annealing)** — one of the most sensitive settings; too fast or too slow an anneal can prevent learning entirely, especially in sparse-reward tasks like RWARE.
2. **Learning rate.**
3. **Target network update interval / rate.**
4. **Replay buffer size** and how many episodes are sampled per update.
5. **Mixing network / hypernetwork capacity.**

**Shared across both:**
- **Discount factor (`gamma`)** — usually fixed near 0.99, but sometimes tuned for very long-horizon sparse-reward tasks.
- **Network architecture size** (hidden dimension, RNN vs. feedforward) — RWARE-style partial observability generally favors keeping recurrence.
- **Random seed** — not a hyperparameter in the tuning sense, but MARL literature consistently emphasizes running many seeds (5–10) since variance across seeds can be large, especially for QMIX-family methods.

**Starting-point recommendation for our tuning work:** given the RWARE-specific literature above, prioritize tuning MAPPO's epoch count and entropy coefficient first (these showed the largest documented swings), and prioritize QMIX's exploration schedule if we decide to keep tuning QMIX rather than defaulting to MAPPO.

---

## 5. Checks to close the [VERIFY] items

Confirm the exact hyperparameter names/defaults actually used in `algorithms/qmix.py`, `algorithms/mappo.py`, `configs/qmix_rware_homogeneous.yaml`, and `configs/mappo_rware_homogeneous.yaml` against this doc's Section 2 (written around EPyMARL, not our custom implementation), and against `MARL_Literature_Review.xlsx` for Section 3. Update this doc with the confirmed values and remove the **[VERIFY]** tags as they are settled.
