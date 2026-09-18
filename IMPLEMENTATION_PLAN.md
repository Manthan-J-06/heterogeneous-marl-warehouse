# Implementation Plan — MARL for Heterogeneous Warehouse Fleets

## 1. Problem statement

Multi-agent reinforcement learning has been applied to warehouse coordination problems —
task allocation, cooperative path finding, order picking — with strong recent results.
However, the large majority of this work assumes a **homogeneous fleet**: every agent in the
environment has identical movement speed, payload capacity, and (where modeled) energy
consumption. This is a simplifying assumption that does not hold in operational warehouses,
where fleets are routinely mixed — different robot models procured at different times,
different capacity classes for different task types, and varying battery health across the
fleet.

This project addresses **parameter-level heterogeneity**: agents of the same general class
(e.g., autonomous ground vehicles) that differ along continuous or discrete parameters —
speed, load capacity, battery/energy budget — rather than differing in kind (e.g., robot vs.
human). It further addresses this jointly with a **multi-objective formulation**, optimizing
for energy efficiency alongside throughput rather than throughput in isolation.

## 2. Literature positioning

A structured review of ten papers spanning MARL warehouse coordination, cooperative path
finding, and adjacent multi-robot energy-aware coordination work (full table:
`MARL_Literature_Review.xlsx`) surfaces a consistent pattern:

- Homogeneous-fleet assumptions are explicit in multiple task-allocation and path-finding
  formulations (QMIX with action masking; HSAC-based dynamic task allocation; constrained
  multi-agent navigation work that explicitly flags heterogeneous-fleet modeling as future
  work).
- The one paper modeling heterogeneity does so at the level of agent *type* (AGV vs. human
  picker, differing action/observation spaces), optimizes pick-rate as a single objective,
  and explicitly identifies multi-objective extensions (energy, travel distance) as open
  work.
- Energy-aware multi-agent coordination has been studied in adjacent domains (e.g.,
  autonomous underwater vehicle fleets) but without a heterogeneous-fleet or warehouse
  formulation.

No reviewed work combines parameter-level fleet heterogeneity with a multi-objective
(throughput + energy) formulation in a warehouse coordination setting. This is the gap this
project targets.

## 3. Research question and hypothesis

**Research question:** In a warehouse coordination task with a parameter-heterogeneous
fleet, does a policy that is explicitly conditioned on per-agent heterogeneity parameters
outperform a heterogeneity-blind policy (same algorithm, same fleet, no access to per-agent
parameters beyond what's incidentally observable) on joint throughput and energy-efficiency
metrics?

**Hypothesis:** Heterogeneity-aware policies will learn to allocate tasks in a
parameter-sensitive way (e.g., routing higher-capacity agents to bulk tasks, avoiding
long-distance dispatch of low-battery agents), producing measurable improvements in energy
efficiency at comparable or improved throughput relative to heterogeneity-blind baselines.

## 4. Technical approach

### 4.1 Environment

Base simulation environment: **TA-RWARE** (or **RWARE** if TA-RWARE proves difficult to
extend within the project timeline). Both are open-source, Gymnasium/PettingZoo-compatible
grid-world warehouse simulators, which makes them straightforward to extend with per-agent
parameters and to integrate with standard MARL training libraries.

Planned environment extensions:
- **Per-agent speed**: variable step cost or movement probability per agent.
- **Per-agent load capacity**: constraints on task/shelf assignment based on agent capacity
  class.
- **Per-agent battery/energy budget**: energy depletion as a function of movement and load,
  with recharging behavior and energy-aware task feasibility constraints.

### 4.2 Reward formulation

A multi-objective reward combining:
- Task completion signal (throughput term, consistent with baseline literature).
- Energy-efficiency term (penalizing unnecessary movement/energy expenditure, consistent with
  energy-aware reward formulations used in adjacent multi-agent energy literature).

Relative weighting between these terms is an experimental variable, not a fixed constant —
sensitivity to this weighting will be part of the evaluation.

### 4.3 Algorithms

Baseline algorithms: **MAPPO** and **QMIX**, both well-established in the reviewed
literature and available in tested open-source implementations (EPyMARL), used to avoid
reimplementation risk in the base algorithm and to keep the experimental focus on the
heterogeneity-awareness variable.

Heterogeneity-aware variant: per-agent parameters (speed, capacity, battery level) included
in the agent's observation/state representation, with architecture options under
consideration including parameter-conditioned policy networks and hypernetwork-based
parameter sharing. Final architecture choice will be made after baseline reproduction,
informed by what the baseline results show about where heterogeneity-blind policies fail.

### 4.4 Experimental design

Three conditions, same underlying algorithm, compared directly:
1. Homogeneous fleet (all agents identical) — reference condition, consistent with prior
   literature results.
2. Heterogeneous fleet, heterogeneity-**blind** policy (agents differ, but the policy has no
   explicit access to per-agent parameters).
3. Heterogeneous fleet, heterogeneity-**aware** policy (agents differ, policy is
   parameter-conditioned).

Varied across: warehouse grid size, agent density, and degree of heterogeneity (parameter
variance across the fleet).

### 4.5 Evaluation metrics

- **Throughput**: tasks completed per unit time.
- **Energy efficiency**: energy consumed per completed task.
- **Task completion time**: distribution, not just mean, to capture tail behavior.
- **Per-agent workload distribution**: to check whether heterogeneity-aware policies produce
  sensible allocation patterns (e.g., not systematically overloading low-battery agents).

## 5. Phased plan

### Phase 1 — Environment setup
Stand up the base simulator, verify it runs end-to-end with a random policy on a small grid,
and confirm logging of the core metrics above. This validates the environment before any
algorithm work begins.

### Phase 2 — Baseline reproduction
Train MAPPO and/or QMIX on the **homogeneous** fleet configuration, verified against
published trends from the reviewed literature. This establishes ground truth: without a
correctly reproduced baseline, any later improvement claim is not credible.

### Phase 3 — Heterogeneous extension
Introduce per-agent parameters into the environment and implement the heterogeneity-aware
policy variant. Implement the heterogeneity-blind heterogeneous-fleet condition as a direct
comparison point.

### Phase 4 — Experiments and ablations
Run the three-condition comparison across multiple grid sizes, densities, and heterogeneity
levels. Include a deliberate stress-test scenario (e.g., a low-battery agent stranded
mid-task, an unusually high-capacity-variance fleet) rather than only well-behaved cases, to
surface failure modes rather than only favorable results.

### Phase 5 — Writing
Consolidate results into a paper draft: problem framing, related work, method, experimental
results, and discussion of limitations. Target venue to be determined based on the strength
of early results.

## 6. Compute

Phase 1–2 (small grid, small agent counts) run on a standard laptop CPU. Phase 3–4 (larger
fleets, more training episodes, hyperparameter sweeps) will use available higher-spec
hardware and college compute resources as needed. All tooling used across every phase is
open-source and free of cost.
