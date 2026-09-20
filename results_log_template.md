# Baseline Training Results — Homogeneous Fleet (QMIX vs MAPPO)

Fill this in after each full training run completes — don't fabricate numbers,
this is meant to be an honest record for the write-up and for comparison
against the literature review.

## Run Configuration

| | QMIX | MAPPO |
|---|---|---|
| Config file | `configs/qmix_rware_homogeneous.yaml` | `configs/mappo_rware_homogeneous.yaml` |
| Env | rware-tiny-4ag-easy-v2 | rware-tiny-4ag-easy-v2 |
| Total env steps | | |
| Wall-clock training time | | |
| Hardware used (CPU/GPU) | | |
| Final epsilon / — | | n/a |
| Learning rate | 0.0005 | 0.0003 |

## Final Metrics

| Metric | QMIX | MAPPO |
|---|---|---|
| Final mean episode reward (last 50 episodes) | | |
| Episode reward at 25% / 50% / 75% / 100% of training | | |
| Training loss trend (converged? still decreasing? diverged?) | | |
| Any instability observed (loss spikes, reward collapse)? | | |

## Comparison Against Literature Review Trends

Reference: `MARL_Literature_Review.xlsx` / `MARL_Warehouse_Literature_Review.md`

- Does the QMIX convergence pattern (rate, stability) plausibly match trends reported in the Choi et al. and other QMIX-based papers reviewed? Yes / No / Partially — explain:
- Does MAPPO show the sample-efficiency or stability advantage over value-decomposition methods that the MAPPO literature (Yu et al.) suggests? Yes / No / Partially — explain:
- Any surprising divergence from expected trends? If so, is it a bug, a hyperparameter issue, or a genuine finding worth investigating further?

## Notes / Issues Encountered

-

## Next Steps

- [ ] Re-run with a second seed to check variance before trusting single-run curves
- [ ] Decide if this baseline is solid enough to move to the heterogeneous-fleet phase
