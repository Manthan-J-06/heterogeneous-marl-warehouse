# Phase 4b: Multi-Fleet Evaluation & Heterogeneity Ablation Results

Commit Evaluated: `6ed32c8`

## 1. Overview & Gate Verification
- **Gate Check**: `.venv/Scripts/python.exe eval_controls.py --fleet_seed 1` passed identically:
  - `uniform`: 33 deliveries over 200 episodes
  - `fwd40`: 62 deliveries over 200 episodes
  - `left40`: 18 deliveries over 200 episodes
- **Capacity Caveat**: Search for `carrying_shelf` in `heterogeneous_env.py` yielded 0 matches. Agent load capacities ($1$ to $8$) do not functionally constrain shelf transport in the current environment simulation.
- **Evaluation Protocol**: Each evaluation spans 4 base seeds (`1000`, `2000`, `3000`, `4000`) $\times$ 50 episodes = 200 total episodes of 500 steps each.

---

## 2. Master Results Table

| Configuration / Fleet | Commit | random (uniform) | blind | aware | fwd40 | left40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fleet 2** (seed 42) | `6ed32c8` | 39 | 13 | 52 | **67** | 15 |
| **Fleet 3** (seed 42) | `6ed32c8` | 40 | 15 | 62 | **70** | 25 |
| **Fleet 4** (seed 42) | `6ed32c8` | 30 | 30 | 38 | **70** | 19 |
| **Fleet 5** (seed 42) | `6ed32c8` | 28 | 13 | 54 | **69** | 21 |
| **Fleet 1 (A4: seed 1)** | `6ed32c8` | 33 | 25 | 39 | **62** | 18 |
| **High Variance Fleet 1 (A3)** | `6ed32c8` | 5 | 5 | 3 | **17** | 3 |

---

## 3. Key Observations & Findings

### A. Aware Policy Never Outperforms Forward-Biased Heuristic (`fwd40`)
Across all evaluated fleets (Fleets 1, 2, 3, 4, 5, and High Variance), `fwd40` (a simple random policy with 40% forward-action bias) consistently achieved higher deliveries than the trained MAPPO `aware` policy. At 30,000 training steps, policy gains primarily reflect learning a coarse forward-motion bias rather than cooperative warehouse routing.

### B. Information Ablation (Zeroed Features Control on Fleet 2)
To isolate whether the aware model's performance reflects genuine feature utilization vs. extra network parameters:
- **Actual Fleet Features**: **61 deliveries**
- **Zeroed Features (`[0, 0, 0]`)**: **55 deliveries**
- **Difference**: 6 deliveries over 200 episodes (~0.03 deliveries/ep).
This demonstrates that feature awareness accounts for minimal behavioral differentiation at this training budget.

### C. Seed Instability (A4)
Training Fleet 1 with seed 1 vs seed 42:
- Seed 42: aware achieved 63 deliveries
- Seed 1: aware achieved **39 deliveries** (38% drop)
This underlines that single-seed training runs carry substantial stochastic variance.

### D. High Variance Bottlenecks (A3)
In wide-variance fleets (`configs/high_variance_fleet.yaml`, speeds down to 0.2):
- Deliveries dropped sharply across all variants (`blind`: 5, `aware`: 3, `fwd40`: 17) due to slow-agent traffic bottlenecks.
