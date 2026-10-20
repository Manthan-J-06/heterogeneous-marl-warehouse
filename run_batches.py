import json
import os
from experiment_runner import run_fleet_pipeline

results = []

# Fleet 2: Evaluated on commit 6ed32c8 (train_seed=42). Note: original pre-PR run recorded
# random=39, blind=13, aware=52, fwd40=67, left40=15. With per-episode torch.manual_seed(s)
# in evaluate(), eval yields random=39, blind=12, aware=50, fwd40=67, left40=15 (noise floor +/-2).
# Kept here as historical baseline / reference run:
results.append({
    "Fleet": "Fleet 2 (pasted from earlier run)",
    "Commit": "6ed32c8",
    "random": 39,
    "blind": 13,
    "aware": 52,
    "fwd40": 67,
    "left40": 15
})

# Run Fleets 3, 4, 5
for f in [3, 4, 5]:
    r = run_fleet_pipeline(f, train_seed=42)
    results.append(r)

# Run A4: Fleet 1 with train_seed 1
r_a4 = run_fleet_pipeline(1, train_seed=1)
results.append(r_a4)

print("\n" + "=" * 50)
print("FINAL RESULTS TABLE")
print("=" * 50)
print("| Fleet | Commit | random | blind | aware | fwd40 | left40 |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in results:
    print(f"| {r['Fleet']} | {r['Commit']} | {r['random']} | {r['blind']} | {r['aware']} | {r['fwd40']} | {r['left40']} |")

with open("logs/full_experiment_results.json", "w") as fp:
    json.dump(results, fp, indent=2)
