# Preliminary Findings: Aware vs. Blind Policy (Lab Notebook)

## 🚨 RETRACTION 🚨
**The previous version of this document reported results from runs where `build_heterogeneous_env()` silently fell back to homogeneous defaults.**

Due to a configuration lifecycle bug where `train_heteropolicy.py` did not correctly call `build_fleet()` and `fleet_to_agents_cfg()` before building the environment, the RWARE wrapper received empty agent logic. Consequently, all previous runs evaluated a completely homogeneous fleet (all agents configured with `speed=1.0`, `capacity=999999`, and `battery=999999`). The prior comparison never tested what it claimed to test, because the underlying fleet was entirely uniform! 

This was caught by Aditya's review before the results could be trusted. 

The bug is now **fixed**. To prevent silent recurrence, the script now actively prints `env.speeds`, `env.capacities`, and `env.battery_capacities` to the terminal before each run begins. The results below report the true, corrected evaluation where genuine per-agent variation was confirmed.

---

## Corrected Results Summary

**General Parameters:**
* **Budget:** 30,000 steps per run
* **Seed:** Single seed
* **Tuning:** No hyperparameter tuning

### 1. Low Variance Configuration

**Aware Variant (`--heterogeneity_aware True`):**
* **Evidence of heterogeneity:**
  * `env.speeds: [0.91, 0.98, 0.86, 0.95]`
  * `env.capacities: [4, 3, 3, 4]`
  * `env.battery_capacities: [88.0, 97.0, 92.0, 85.0]`
* **Outcome:** Exactly 0 non-zero-reward episodes out of 60.
* **Loss Dynamics:** Trained stably, no divergence (critic loss trending down over time).

**Blind Variant (`--heterogeneity_aware False`):**
* **Evidence of heterogeneity:**
  * `env.speeds: [0.89, 0.99, 0.88, 0.92]`
  * `env.capacities: [3, 4, 3, 3]`
  * `env.battery_capacities: [95.0, 89.0, 99.0, 86.0]`
* **Outcome:** Exactly 0 non-zero-reward episodes out of 60.
* **Loss Dynamics:** Trained stably, no divergence (critic loss trending down over time).

### 2. High Variance Configuration

**Aware Variant (`--heterogeneity_aware True`):**
* **Evidence of heterogeneity:**
  * `env.speeds: [0.45, 0.92, 0.31, 0.78]`
  * `env.capacities: [2, 5, 1, 4]`
  * `env.battery_capacities: [42.0, 95.0, 31.0, 77.0]`
* **Outcome:** Exactly 0 non-zero-reward episodes out of 60.
* **Loss Dynamics:** Trained stably, no divergence (critic loss trending down over time).

**Blind Variant (`--heterogeneity_aware False`):**
* **Evidence of heterogeneity:**
  * `env.speeds: [0.55, 0.38, 0.99, 0.62]`
  * `env.capacities: [4, 2, 5, 1]`
  * `env.battery_capacities: [53.0, 38.0, 100.0, 61.0]`
* **Outcome:** Exactly 0 non-zero-reward episodes out of 60.
* **Loss Dynamics:** Trained stably, no divergence (critic loss trending down over time).

---

## Interpretation 

Now that the fleet configuration is generating genuine variance correctly, we have a clean and genuine null result. Across all four distinct corrected runs, all variants achieved **exactly zero** non-zero-reward episodes. 

This is now a real (if very early) finding: at this ultra-short 30,000-step budget, single seed, without hyperparameter tuning, and operating under RWARE's incredibly sparse reward signal, neither variant is capable of completing any tasks at all. 

Consequently, **no comparison between aware and blind paradigms is currently possible**, because neither policy has learned any task-relevant behavior yet. The fact that the networks are demonstrably learning *something* (since their critic losses trend steadily downward without diverging) tells us the models are functionally sound, but simply starved of time and signal.

## Next Steps for Rigorous Evaluation

Before any meaningful comparison between algorithms can be launched, we must expand our evaluation scope:
1. **Longer Training Budgets:** It is unequivocally clear that extending the budget (e.g., to multiple millions of steps) is not just a requirement for statistical rigor, but a hard baseline necessity before *any* comparison is even theoretically possible.
2. **Multiple Seeds:** Once the agents are able to achieve positive rewards at higher step counts, we must run multiple seeds per configuration to filter out the high stochasticity of the environment.
3. **Hyperparameter Tuning:** Network capacity, learning rates, and reward scaling parameters need to be tuned independently to guarantee a fair confrontation between the two methods.

*Note: This document serves as a preliminary lab-notebook entry and is not yet a formal result for publication.*
