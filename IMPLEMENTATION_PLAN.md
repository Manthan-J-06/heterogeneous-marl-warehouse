# Implementation Plan — MARL for Heterogeneous Warehouse Fleets

**Status:** Draft for team review — comments/changes welcome before we lock this in.

## 1. The problem, in one paragraph

Multi-agent reinforcement learning has been applied to warehouse coordination (task
allocation, path finding, order picking), but almost every paper in our literature review
assumes a **homogeneous fleet** — every agent has the same speed, capacity, and battery/energy
profile. Several papers explicitly flag this as a limitation or as future work. The one paper
that does model heterogeneity (Krnjaic et al., 2024) does so at the level of *agent type*
(AGVs vs. human pickers) and optimizes pick rate only, explicitly leaving multi-objective
optimization (energy, travel distance) as open work.

## 2. Our angle

We model **parameter-heterogeneous** fleets — agents of the same general type (e.g., AGVs)
that differ in speed, load capacity, and battery/energy consumption — and optimize for
**throughput and energy efficiency jointly**, not throughput alone.

**Core research question:** Does a policy that is explicitly aware of per-agent heterogeneity
(vs. one that treats all agents as interchangeable) improve both task throughput and energy
efficiency in a warehouse coordination setting?

**Why this is novel:** no paper in our review models this combination. It's a controlled,
answerable question — we can run the same algorithm on a homogeneous fleet, a heterogeneous
fleet with a heterogeneity-*blind* policy, and a heterogeneous fleet with a heterogeneity-*aware*
policy, and compare.

## 3. Related work — where we sit

| Paper | Heterogeneity | Objective | Gap we're addressing |
|---|---|---|---|
| Choi et al. 2022 (QMIX + masking) | None (homogeneous) | Throughput | No heterogeneity |
| Tang et al. 2021 (HSAC) | None (homogeneous) | Throughput | No heterogeneity |
| Krnjaic et al. 2024 | Agent-*type* (AGV vs. human) | Pick rate only | We go parameter-level + multi-objective |
| Gao & Prorok 2023 | None (flagged as future work) | Path efficiency (SPL) | No heterogeneity |
| Wibisono et al. 2025 (AUV survey) | None (agents identical) | Energy (single-objective) | Different domain (underwater), no heterogeneity |

Full detail lives in `MARL_Literature_Review.xlsx` in this repo.

## 4. Phases

We are **not** starting implementation until the team has reviewed and discussed this
document. Each phase below ends with something concrete the whole team can look at — not
just a status update.

### Phase 0 — Scoping (this document)
Team reviews this plan, raises objections/changes. Locked once we agree.

### Phase 1 — Environment setup (Week 1 target)
- Stand up TA-RWARE (or RWARE, if TA-RWARE proves too heavy to modify quickly) as our base
  simulator — it's open-source, lightweight, and already warehouse-shaped.
- Get a small grid (small agent count) running end-to-end: reset, step, random-policy rollout.
- **Deliverable:** a script that runs N random-policy episodes and prints/logs reward,
  pick-rate, and step count. This is the "show progress" artifact for week 1 — it doesn't need
  to be the real algorithm yet, it needs to prove the environment works.

### Phase 2 — Baseline reproduction
- Get a known algorithm (MAPPO or QMIX) training on the **homogeneous** version of the
  environment, verified against published numbers/trends from the papers above.
- This is our ground truth — without a working, verified baseline, we can't credibly claim
  our heterogeneous extension improves anything.

### Phase 3 — Heterogeneous extension (core contribution)
- Introduce per-agent parameters: speed, load capacity, battery/energy budget.
- Build the heterogeneity-aware variant (policy/observation/reward design changes — exact
  mechanism TBD as a team decision once Phase 2 baseline is solid).
- Add the energy-efficiency term to the reward/evaluation.

### Phase 4 — Experiments
- Homogeneous baseline vs. heterogeneous-blind vs. heterogeneous-aware, across a few
  warehouse sizes/densities.
- Ablations on which parameter(s) of heterogeneity matter most.
- Deliberately test an edge case likely to break something (e.g., a low-battery agent
  stranded mid-task) rather than only the easy cases.

### Phase 5 — Writing
- Target venue TBD as a team decision once we see early results — a workshop paper is a
  reasonable stretch goal; worst case this is still a strong course/portfolio deliverable
  regardless of where it's submitted.

## 5. Open questions for team discussion today

1. Does everyone agree on parameter-heterogeneity (speed/capacity/battery) over
   agent-type-heterogeneity (AGV vs. human)? See prior discussion — parameter-level is lower
   engineering risk and still genuinely open.
2. Any objections to TA-RWARE as the base environment, or does anyone have experience with a
   different MARL warehouse sim we should consider instead?
3. Algorithm choice for the baseline — MAPPO vs. QMIX. Whoever's more comfortable
   implementing/debugging one of these should probably say so now.

## 6. Compute & cost

Phase 1–2 run fine on a laptop CPU at small scale. Phase 3–4 (larger fleets, more training
episodes) will lean on whoever has the stronger laptop and/or the college's compute resources.
Everything in this plan — the environment, the algorithms, the tooling — is open-source and
free; no paid services are required. If that ever changes, it'll be flagged before we commit
to it.
