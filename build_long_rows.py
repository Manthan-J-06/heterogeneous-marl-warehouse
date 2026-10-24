"""Turn seed_job.py result files into rows that aggregate_results.py can read.

Usage:
  python build_long_rows.py "<glob>" <random> <fwd40> <left40> [label] > logs/rows.json
  python aggregate_results.py logs/rows.json

Example (500k steps, Fleet 2; controls from eval_controls.py):
  python build_long_rows.py "logs/long500000_fleet2_*_seed*.json" 39 67 15 "Fleet 2" > logs/rows_500k.json

Example (tight battery, 300k steps):
  python build_long_rows.py "logs/long300000_tight_battery_fleet_fleet2_*_seed*.json" 10 31 7 "Fleet 2" > logs/rows_tight.json

Each result file is {"fleet_seed", "mode", "train_seed", "total", ...}. A training seed is
kept only if it has both a blind and an aware file; others are reported on stderr.
"""
import glob
import json
import sys

if len(sys.argv) < 5:
    sys.exit(__doc__)

pattern = sys.argv[1]
random_, fwd40, left40 = (int(x) for x in sys.argv[2:5])
label = sys.argv[5] if len(sys.argv) > 5 else None

runs = {}
for path in sorted(glob.glob(pattern)):
    d = json.load(open(path))
    runs.setdefault((d["fleet_seed"], d["train_seed"]), {})[d["mode"]] = d["total"]

if not runs:
    sys.exit(f"No files matched: {pattern}")

rows = []
for (fleet, seed), m in sorted(runs.items()):
    if "blind" not in m or "aware" not in m:
        print(f"skipping fleet {fleet} seed {seed}: only has {list(m)}", file=sys.stderr)
        continue
    rows.append({
        "Fleet": f"{label or f'Fleet {fleet}'} (seed {seed})",
        "Commit": "local",
        "random": random_,
        "blind": m["blind"],
        "aware": m["aware"],
        "fwd40": fwd40,
        "left40": left40,
    })

print(json.dumps(rows, indent=1))
