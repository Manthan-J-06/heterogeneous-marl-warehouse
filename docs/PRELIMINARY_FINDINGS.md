# Preliminary Findings: Aware vs. Blind Policy (Lab Notebook)

**Budgets:** 30,000 steps per run
**Seed:** Single seed
**Tuning:** No hyperparameter tuning

## Results Summary: Low Variance Configuration

**Configuration:** `configs/low_variance_fleet.yaml`

* **Aware Variant (`--heterogeneity_aware True`):**
  * Achieved 1 non-zero-reward episode out of 60.
  * In Episode 28, it achieved a mean reward of 0.250.
  * Critic loss trend: Stable, non-diverging (average trend ~0.045 → ~0.008 over time).

* **Blind Variant (`--heterogeneity_aware False`):**
  * Achieved 0 non-zero-reward episodes out of 60.
  * Critic loss trend: Stable, non-diverging (average trend ~0.005 → ~0.0008 over time).

## Results Summary: High Variance Configuration

**Configuration:** `configs/high_variance_fleet.yaml`

* **Aware Variant (`--heterogeneity_aware True`):**
  * Achieved 0 non-zero-reward episodes out of 60.

* **Blind Variant (`--heterogeneity_aware False`):**
  * Achieved 1 non-zero-reward episode out of 60.
  * In Episode 1, it achieved a mean reward of 0.250.

## Combined Interpretation and Caveats

Across both configurations tested (low-variance and high-variance), the aware variant has exactly 1 total task-completion event, and the blind variant also has exactly 1 total task-completion event. 

These two single-seed, short-budget runs show **no consistent directional advantage** for heterogeneity-awareness so far. If anything, the results are entirely consistent with pure random chance, given how rare task completions are under RWARE's sparse reward schema with an early-stage policy. There is no trend or advantage in either direction to claim at this juncture.

However, all variants exhibited stable and non-diverging training dynamics (evidenced by decreasing critic losses), which confirms the soundness of the base implementation and allows us to proceed to larger-scale evaluations.

## Next Steps for Rigorous Evaluation

Given this strongly null-ish preliminary result, drawing a genuine conclusion about the effectiveness of heterogeneity-aware policies absolutely requires the following next steps:
1. **Multiple Seeds:** It is critical to run experiments across multiple seeds per variant to establish statistical significance. Sparse rewards heavily skew single-seed comparisons.
2. **Longer Training Budgets:** 30,000 steps is far too short to observe meaningful policy convergence in RWARE. Budgets in the millions of steps are required for a proper evaluation.
3. **Hyperparameter Tuning:** Proper tuning of learning rates, network dimensions, and advantage computations must be conducted for both models to capture their actual learning capacities.

*Note: This document serves as a preliminary lab-notebook entry and is not yet a formal result for publication.*
