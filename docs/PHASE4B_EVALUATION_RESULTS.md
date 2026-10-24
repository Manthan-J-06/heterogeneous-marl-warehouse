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
  - **Does capacity bite?** The *block counts* are large, but the *delivery effect is not established*:
    - *Under MAPPO Aware policy (Fleet 2 checkpoint, 200 episodes)*: capacity 1 produced 81,473 capacity blocks (`capacity_blocks_this_step`) and 42 deliveries; capacity 8 produced 8,019 blocks and 57 deliveries.
    - *Under `fwd40` control policy (Fleet 1, 200 episodes)*: capacity 1 produced 108,740 blocks (47 deliveries) vs. 39,267 blocks (71 deliveries) at capacity 8.
    - With counts this small, run-to-run noise on deliveries is about the square root of the total (roughly +/-7 at 50 deliveries, +/-11 on a difference of two such counts), so 42 vs 57 is at most borderline and should not be called significant. Each comparison is also a single checkpoint/fleet.
    - Block counts are inflated by the *phantom carrying* artefact above: any accidental TOGGLE flips `is_carrying`, so an agent can be "carrying" nothing and still be blocked. The numbers measure how often the proxy fires, not real physical load.
    - Provenance: the block counts come from an ad-hoc capacity sweep whose script was not committed, so they cannot be regenerated from this repo. Treat them as indicative only.

---

## 2. Master Results Table

| Configuration / Fleet | Commit | random (uniform) | blind | aware | fwd40 | left40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fleet 2** (seed 42) *(historical, not reproducible - see note)* | `6ed32c8` | 39 | 13 | 52 | **67** | 15 |
| **Fleet 3** (seed 42) | `6ed32c8` | 40 | 15 | 62 | **70** | 25 |
| **Fleet 4** (seed 42) | `6ed32c8` | 30 | 30 | 38 | **70** | 19 |
| **Fleet 5** (seed 42) | `6ed32c8` | 28 | 13 | 54 | **69** | 21 |
| **Fleet 1 (A4: seed 1)** | `6ed32c8` | 33 | 25 | 39 | **62** | 18 |

*Single training seed per row at 30k steps: treat every row as indicative only. Multi-seed results are in Section 4.*
| **High Variance Fleet 1 (A3)** | `6ed32c8` | 5 | 5 | 3 | **17** | 3 |

*Note on Fleet 2 provenance* (the 52 above, and the 63 in Section 3C, come from earlier unseeded runs and cannot be regenerated; the re-evaluated 50 is the reproducible figure): The Fleet 2 row above represents the historical run from the initial batch (`random`: 39, `blind`: 13, `aware`: 52). When re-evaluated with deterministic per-episode torch seeding (`torch.manual_seed(s)`), evaluation yields `random`: 39, `blind`: 12, `aware`: 50, `fwd40`: 67, `left40`: 15, confirming the evaluation noise floor of approximately $\pm 2$ deliveries across runs on the same checkpoint.

---

## 3. Key Observations & Findings

### A. Aware Policy vs. Forward-Biased Heuristic (`fwd40`)
At 30k steps and a single training seed, across all evaluated fleets (Fleets 1, 2, 3, 4, 5, and High Variance), `fwd40` (a simple random policy with 40% forward-action bias) consistently achieved higher deliveries than the trained MAPPO `aware` policy. At 30,000 training steps, we hypothesize that policy gains may largely reflect learning a coarse forward-motion bias rather than coordinated multi-agent routing. This is a 30k-step statement only: at 500k steps (Section 4) trained policies are compared against the controls separately. Full validation would require action distribution histograms or specialized diagnostic probes.

### B. Information Ablation (Zeroed Features Control on Fleet 2)
To investigate whether the aware model's performance reflects genuine feature utilization vs. extra network parameters, we evaluated the Fleet 2 aware checkpoint with actual fleet features vs. zeroed features (`[0, 0, 0]`) under deterministic per-episode seeding:
- **Actual Fleet Features**: **50 deliveries**
- **Zeroed Features (`[0, 0, 0]`)**: **60 deliveries**
- **Difference**: zeroed features scored 10 *more* deliveries over 200 episodes (within noise).
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

---

## 4. Follow-up Multi-Seed Results (supersedes single-seed numbers above)

These were run after this document's 30k-step batch, with 3-5 training seeds per mode and the same 200-episode protocol (scripts: `seed_job.py`, `build_long_rows.py`, `aggregate_results.py`, `check_checkpoints.py`):

- **500k steps, Fleet 2 (3 training seeds):** random 39, blind 94.3 ± 11.6, aware 86.3 ± 17.0, fwd40 67, left40 15.
  Aware minus blind is -8.0 (z = -0.6, within noise), and aware beat blind in only 1 of 3 seeds. Aware minus fwd40 is +19.3 (z = +1.6, within noise).
  Both trained policies are well above uniform random, but aware and blind are a tie.
- **Tight-battery fleet, 300k steps, Fleet 2 (3 training seeds):** random 10, blind 16.7 ± 9.3, aware 23.3 ± 19.1, fwd40 31, left40 7.
  Aware minus blind is +6.7 (z = +1.1, within noise), and aware beat blind in 2 of 3 seeds. Aware minus fwd40 is -7.7 (z = -1.0, within noise).
  This is a tie too, with the opposite sign to 500k.
- **Checkpoint diagnostic:** evaluating aware checkpoints with real, zeroed and rotated robot numbers gives about the same deliveries, i.e. the aware policy does not use the robot numbers. Reason: the agent ID is already an input to both policies and the fleet is fixed, so the extra inputs are redundant.

**Exact totals (deliveries over 200 episodes; mean ± spread across training seeds 11, 12, 13):**

| Setting | random | blind | aware | fwd40 | left40 | aware-blind (z) | aware-fwd40 (z) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 500k steps, Fleet 2 | 39 | 94.3 ± 11.6 | 86.3 ± 17.0 | 67 | 15 | -8.0 (-0.6, within noise); aware > blind in 1/3 seeds | +19.3 (+1.6, within noise) |
| Tight battery, 300k steps, Fleet 2 | 10 | 16.7 ± 9.3 | 23.3 ± 19.1 | 31 | 7 | +6.7 (+1.1, within noise); aware > blind in 2/3 seeds | -7.7 (-1.0, within noise) |

Both trained policies learn well above uniform random at 500k, but neither is clearly better than the forward-biased control, and aware vs. blind is a tie in both settings.
