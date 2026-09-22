# Preliminary Findings: Aware vs. Blind Policy (Lab Notebook)

**Configuration:** `configs/low_variance_fleet.yaml`
**Budget:** 30,000 steps
**Seed:** Single seed
**Tuning:** No hyperparameter tuning

## Results Summary

* **Aware Variant (`--heterogeneity_aware True`):**
  * Achieved 1 non-zero-reward episode out of 60.
  * In Episode 28, it achieved a mean reward of 0.250.
  * Critic loss trend: Stable, non-diverging (average trend ~0.045 → ~0.008 over time).

* **Blind Variant (`--heterogeneity_aware False`):**
  * Achieved 0 non-zero-reward episodes out of 60.
  * Critic loss trend: Stable, non-diverging (average trend ~0.005 → ~0.0008 over time).

## Interpretation and Caveats

This is a single-seed, short-budget preliminary result and is **NOT statistically meaningful** on its own. 

Task completion under RWARE's sparse reward structure is rare and highly stochastic, even under policies that are actively improving. In this context, a single non-zero reward event is fundamentally indistinguishable from random chance. 

However, both variants exhibited stable and non-diverging training dynamics (evidenced by the decreasing critic loss), which confirms the soundness of the base implementations.

## Next Steps for Rigorous Evaluation

To draw a genuine, real conclusion about the effectiveness of heterogeneity-aware policies, the following steps are necessary:
1. **Multiple Seeds:** Run experiments across multiple seeds per variant to establish statistical significance.
2. **Longer Training Budgets:** 30,000 steps is too short to observe meaningful convergence in RWARE; much longer training budgets are required.
3. **High Variance Configurations:** Test the policies under the `high_variance_fleet.yaml` configuration, where the performance gap between aware and blind variants is expected to be substantially more pronounced.

*Note: This document serves as a preliminary lab-notebook entry and is not yet a formal result for publication.*
