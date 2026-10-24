"""Aggregate experiment_runner result rows into per-fleet stats.

Input: one or more JSON files, each a list of rows like
  {"Fleet": "Fleet 2", "Commit": "abc1234", "random": 39, "blind": 13,
   "aware": 52, "fwd40": 67, "left40": 15}
Rows from experiment_runner.py name the training seed as "Fleet N (seed S)";
a plain "Fleet N" means train_seed 42. Values are total deliveries over
EPISODES episodes (default 200).

Usage:
  python aggregate_results.py logs/full_experiment_results.json [more.json ...]

Noise model (rough): deliveries are counts, so sd(total) ~ sqrt(total) and
sd(a - b) ~ sqrt(a + b). This covers evaluation noise only. Training-seed
variance is estimated from the spread across seeds, which needs >= 2 seeds.
"""
import json, math, re, sys
from collections import defaultdict
from statistics import mean, stdev

EPISODES = 200
CONDS = ["random", "blind", "aware", "fwd40", "left40"]


def parse_fleet(label):
    m = re.match(r"\s*Fleet\s+(\d+)(?:\s*\(seed\s+(\d+)\))?", str(label))
    if not m:
        return str(label), 42
    return f"Fleet {m.group(1)}", int(m.group(2) or 42)


def z(a, b):
    d = math.sqrt(a + b)
    return (a - b) / d if d else 0.0


def verdict(zv):
    return "within noise" if abs(zv) < 2 else ("higher" if zv > 0 else "lower")


def main(paths):
    rows = []
    for p in paths:
        rows.extend(json.load(open(p)))
    groups = defaultdict(list)
    for r in rows:
        fleet, seed = parse_fleet(r.get("Fleet") or r.get("Config"))
        groups[fleet].append((seed, r))

    print(f"Totals are deliveries over {EPISODES} episodes. z = (a-b)/sqrt(a+b), |z|<2 = within eval noise.\n")
    print("| Fleet | train seeds | " + " | ".join(CONDS) + " | aware-blind (z) | aware-fwd40 (z) |")
    print("| :--- | :---: | " + " | ".join([":---:"] * len(CONDS)) + " | :---: | :---: |")
    for fleet in sorted(groups):
        items = sorted(groups[fleet], key=lambda x: x[0])
        seeds = [s for s, _ in items]
        cells = []
        for c in CONDS:
            vals = [r[c] for _, r in items]
            if len(vals) > 1:
                cells.append(f"{mean(vals):.1f} ± {stdev(vals):.1f}")
            else:
                cells.append(str(vals[0]))
        aw = mean(r["aware"] for _, r in items)
        bl = mean(r["blind"] for _, r in items)
        fw = mean(r["fwd40"] for _, r in items)
        zb, zf = z(aw, bl), z(aw, fw)
        # With >= 2 seeds, also use the across-seed spread (Welch t on aware vs blind).
        # This captures training-seed variance, which the Poisson z above does not.
        seed_note = ""
        if len(items) >= 2:
            av = [r["aware"] for _, r in items]
            bv = [r["blind"] for _, r in items]
            se = math.sqrt(stdev(av) ** 2 / len(av) + stdev(bv) ** 2 / len(bv))
            t = (mean(av) - mean(bv)) / se if se else 0.0
            wins = sum(1 for a_, b_ in zip(av, bv) if a_ > b_)
            seed_note = f" seed-t={t:+.1f}; aware>blind in {wins}/{len(av)} seeds"
        print(f"| {fleet} | {','.join(map(str, seeds))} | " + " | ".join(cells)
              + f" | {aw - bl:+.1f} ({zb:+.1f}, {verdict(zb)}){seed_note} | {aw - fw:+.1f} ({zf:+.1f}, {verdict(zf)}) |")
    n_seeds = max(len(v) for v in groups.values())
    if n_seeds < 2:
        print("\nNOTE: only one training seed per fleet. Training-seed variance is NOT captured;"
              " treat any aware-vs-blind gap as unproven (Fleet 1 aware moved 63 -> 39 across seeds in PR #20).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
