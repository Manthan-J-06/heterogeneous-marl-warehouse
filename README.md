# Heterogeneous-Fleet Multi-Agent Reinforcement Learning for Warehouse Optimization

## Overview

This project investigates multi-agent reinforcement learning (MARL) for coordinating
**heterogeneous** robotic fleets in automated warehouse environments. Existing MARL-based
warehouse coordination research — task allocation, order picking, path planning — has
predominantly assumed **homogeneous fleets**, where every agent shares identical speed,
payload capacity, and energy characteristics. This assumption breaks down in real deployments,
where fleets mix agents of varying capability by design (different robot models, ages,
battery health, load ratings).

This project models fleets with explicit **per-agent heterogeneity** — variable speed,
load capacity, and battery/energy budget — and studies whether policies that are aware of
this heterogeneity outperform heterogeneity-blind policies on both **task throughput** and
**energy efficiency**, evaluated jointly rather than optimizing throughput alone.

## Research gap

A structured review of prior work (`MARL_Literature_Review.xlsx`) identifies a consistent
gap: heterogeneous fleet modeling is either absent or explicitly deferred to future work
across the major MARL-warehouse papers surveyed (task-allocation, path-finding, and
order-picking formulations alike). The closest prior work models heterogeneity at the level
of *agent type* (e.g., AGV vs. human picker) and optimizes a single throughput metric; it
explicitly leaves multi-objective optimization (energy, travel efficiency) as open work.
This project addresses **parameter-level heterogeneity** combined with a **multi-objective
formulation** (throughput + energy), a combination not addressed in the surveyed literature.

## Research question

Does a heterogeneity-aware coordination policy outperform a heterogeneity-blind policy —
on both throughput and energy efficiency — when the underlying fleet is parameter-heterogeneous?

## Key features (planned)

- **Configurable heterogeneous fleet simulation**: per-agent speed, load capacity, and
  battery/energy consumption profiles, layered on a warehouse grid environment.
- **Multi-objective reward formulation**: joint optimization of task throughput and energy
  efficiency, rather than a single scalar throughput objective.
- **Heterogeneity-aware policy architecture**: agent-parameter-conditioned policies
  (architecture to be finalized during Phase 3 — see `IMPLEMENTATION_PLAN.md`), compared
  against heterogeneity-blind baselines using the same underlying algorithm.
- **Baseline reproduction**: verified MAPPO/QMIX baselines on the homogeneous-fleet case
  before any heterogeneous extension is introduced, to isolate the effect of heterogeneity
  awareness from confounds in the base algorithm.
- **Controlled ablation framework**: homogeneous fleet vs. heterogeneous fleet
  (heterogeneity-blind) vs. heterogeneous fleet (heterogeneity-aware), across multiple
  warehouse sizes and agent densities.
- **Evaluation suite**: throughput, energy consumption, task completion time, and per-agent
  workload distribution, logged consistently across all experimental conditions.

## Tech stack

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.10+ | |
| Simulation environment | TA-RWARE / RWARE | Open-source, Gymnasium/PettingZoo-compatible warehouse grid environment |
| RL framework | PyTorch | |
| MARL algorithm implementations | EPyMARL (or custom MAPPO/QMIX) | Provides tested MAPPO, QMIX, IPPO implementations as a baseline foundation |
| Experiment tracking | TensorBoard | Local, free, no external service dependency |
| Configuration | YAML-based experiment configs | Reproducibility across runs |
| Analysis / plotting | matplotlib, seaborn, pandas | |
| Version control | Git / GitHub | |

All tooling is open-source and free; no paid infrastructure is required at the current scope.

## Repository structure

```
.
├── IMPLEMENTATION_PLAN.md      # detailed technical plan, phases, and methodology
├── MARL_Literature_Review.xlsx # structured literature review and gap analysis
├── src/
│   ├── env/                    # environment wrappers, heterogeneity parameterization
│   ├── agents/                 # policy architectures (baseline + heterogeneity-aware)
│   ├── training/                # training loops, config-driven experiment runner
│   └── evaluation/              # metrics, ablation runners
├── configs/                    # YAML experiment configurations
├── experiments/                # experiment outputs, logs
└── docs/                       # additional writeups, figures for the paper
```

## Status

Environment setup and baseline reproduction (Phases 1–2 of `IMPLEMENTATION_PLAN.md`) are in
progress. See that document for full methodology, evaluation metrics, and the current phase.

## Phase 2 — Baseline Training Pipeline (Homogeneous Fleet)

See `algorithms/`, `envs/`, `configs/`, `train_qmix.py`, `train_mappo.py` for
the QMIX and MAPPO baseline implementations on the homogeneous RWARE fleet.

**Running it:**
\`\`\`bash
python train_qmix.py  --config configs/qmix_rware_homogeneous.yaml
python train_mappo.py --config configs/mappo_rware_homogeneous.yaml
tensorboard --logdir runs/
\`\`\`

**What was verified before this was merged:** environment wrapper, both
training loops, checkpointing, and TensorBoard logging were run end-to-end
on a short step budget. Full 500k-step training runs were not yet executed —
see `results_log_template.md` for the write-up once they are.