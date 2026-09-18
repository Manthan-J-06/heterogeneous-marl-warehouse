# MARL for Heterogeneous Warehouse Fleets

Multi-agent reinforcement learning for warehouse coordination with heterogeneous agents
(varying speed, capacity, and energy budgets), optimizing for throughput **and** energy
efficiency jointly.

## Status

Early-stage — see [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) for the full plan,
current phase, and open decisions the team is discussing.

## Why this project

Existing MARL-for-warehouse work almost universally assumes a homogeneous fleet. We model
per-agent heterogeneity explicitly and test whether heterogeneity-aware coordination improves
both throughput and energy efficiency over treating all agents as interchangeable. Background
and literature review: [`MARL_Literature_Review.xlsx`](./MARL_Literature_Review.xlsx).

## Repo structure

```
.
├── IMPLEMENTATION_PLAN.md   # full plan, phases, open questions — read this first
├── CONTRIBUTING.md          # branching/PR workflow for the team
├── MARL_Literature_Review.xlsx
├── src/                     # code goes here once Phase 1 starts
├── experiments/             # experiment configs/results
└── docs/                    # any additional writeups
```

## Setup

_To be filled in once Phase 1 (environment setup) is underway._

## Team

5 members — see `IMPLEMENTATION_PLAN.md` for phase ownership.

## License / cost

This project uses only open-source, free tooling. No paid services required.
