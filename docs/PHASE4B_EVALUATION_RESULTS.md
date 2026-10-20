# Phase 4b: Multi-Fleet Evaluation & Heterogeneity Ablation Results

Commit Evaluated: `6ed32c8`

## 1. Overview & Experimental Context
- **Experimental Setup & Environment**:
  - Environment: `rware-tiny-4ag-easy-v2` (small warehouse grid with 4 agents and delivery queue size of 8).
  - Training Budget: 30,000 environment steps per model.
  - Training Seed: Seed 42 for Fleets 2, 3, 4, 5, and High Variance Fleet 1; Seed 1 for Fleet 1 (A4). Exactly one training seed per fleet.
  - Battery Configuration: Battery limits temporarily loosened 10x (`[850, 1000]`) pending full recharge station mechanics.
- **Evaluation Protocol**:
  - Each evaluation spans 4 base seeds (`1000`, `2000`, `3000`, `4000`) $\times$ 50 episodes = 200 total episodes of 500 steps each.
  - Sampling is deterministically seeded per episode (`random.seed(s)`, `np.random.seed(s)`, `torch.manual_seed(s)`).
- **Gate Check**: `.venv/Scripts/python.exe eval_controls.py --fleet_seed 1` passed identically:
  - `uniform`: 33 deliveries over 200 episodes
  - `fwd40`: 62 deliveries over 200 episodes
  - `left40`: 18 deliveries over 200 episodes
- **Capacity Caveat & Impact**:
  - `is_carrying` flips on any action 4 (`TOGGLE_LOAD`), whether or not the agent is actually over a shelf, and never queries RWARE's underlying physical shelf state.
  - Consequently, agent load capacity acts strictly on a *simulated carry state* rather than true physical shelf load.
  - **Does capacity bite?** Yes, significantly:
    - *Under MAPPO Aware policy (Fleet 2 checkpoint, 200 episodes)*: At capacity 1, the wrapper injected **81,473 capacity blocks** (`capacity_blocks_this_step`), yielding 42 deliveries. At capacity 8, capacity blocks dropped 10-fold to **8,019**, yielding 57 deliveries.
    - *Under `fwd40` control policy (Fleet 1, 200 episodes)*: Capacity 1 triggered **108,740 blocks** (47 deliveries) vs. **39,267 blocks** at capacity 8 (71 deliveries).

---

## 2. Master Results Table

| Configuration / Fleet | Commit | random (uniform) | blind | aware | fwd40 | left40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fleet 2** (seed 42) *(pasted from earlier run)* | `6ed32c8` | 39 | 13 | 52 | **67** | 15 |
| **Fleet 3** (seed 42) | `6ed32c8` | 40 | 15 | 62 | **70** | 25 |
| **Fleet 4** (seed 42) | `6ed32c8` | 30 | 30 | 38 | **70** | 19 |
| **Fleet 5** (seed 42) | `6ed32c8` | 28 | 13 | 54 | **69** | 21 |
| **Fleet 1 (A4: seed 1)** | `6ed32c8` | 33 | 25 | 39 | **62** | 18 |
| **High Variance Fleet 1 (A3)** | `6ed32c8` | 5 | 5 | 3 | **17** | 3 |

*Note on Fleet 2 provenance*: The Fleet 2 row above represents the historical run from the initial batch (`random`: 39, `blind`: 13, `aware`: 52). When re-evaluated with deterministic per-episode torch seeding (`torch.manual_seed(s)`), evaluation yields `random`: 39, `blind`: 12, `aware`: 50, `fwd40`: 67, `left40`: 15, confirming the evaluation noise floor of approximately $\pm 2$ deliveries across runs on the same checkpoint.

---

## 3. Key Observations & Findings

### A. Aware Policy vs. Forward-Biased Heuristic (`fwd40`)
Across all evaluated fleets (Fleets 1, 2, 3, 4, 5, and High Variance), `fwd40` (a simple random policy with 40% forward-action bias) consistently achieved higher deliveries than the trained MAPPO `aware` policy. At 30,000 training steps, we hypothesize that policy gains may largely reflect learning a coarse forward-motion bias rather than coordinated multi-agent routing. Full validation would require action distribution histograms or specialized diagnostic probes.

### B. Information Ablation (Zeroed Features Control on Fleet 2)
To investigate whether the aware model's performance reflects genuine feature utilization vs. extra network parameters, we evaluated the Fleet 2 aware checkpoint with actual fleet features vs. zeroed features (`[0, 0, 0]`) under deterministic per-episode seeding:
- **Actual Fleet Features**: **50 deliveries**
- **Zeroed Features (`[0, 0, 0]`)**: **60 deliveries**
- **Difference**: -10 deliveries over 200 episodes.
Conclusion: There is **no detectable reliance on fleet features at this budget (noise about $\pm 11$ on differences)**. Inference-time feature zeroing also cannot explain the training-time performance gap between blind (12–13) and aware (50–52); testing that will require multiple training seeds across both conditions or training aware models on zeroed/shuffled features.

### C. Seed Instability (A4)
Training Fleet 1 with seed 1 vs seed 42:
- Seed 42: aware achieved 63 deliveries *(pasted from an earlier run on commit `6ed32c8`)*
- Seed 1: aware achieved **39 deliveries** (38% drop)
This underlines that single-seed training runs carry substantial stochastic variance at 30k steps.

### D. High Variance Fleet Behavior (A3)
In wide-variance fleets (`configs/high_variance_fleet.yaml`, speeds down to 0.2):
- Deliveries dropped sharply across all variants (`blind`: 5, `aware`: 3, `fwd40`: 17, `uniform`: 5).
- Note that `blind` (5), `aware` (3), and `uniform` (5) are statistically indistinguishable within evaluation noise ($\approx \pm 2$ to $3$ deliveries).
- The drop in deliveries is hypothesized to arise from slow-agent traffic bottlenecks and reduced overall fleet motion, though isolating this requires specific congestion and trajectory metrics.
