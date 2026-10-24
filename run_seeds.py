import json, os
from experiment_runner import run_fleet_pipeline
out = "logs/seed_rows.json"
os.makedirs("logs", exist_ok=True)
rows = json.load(open(out)) if os.path.exists(out) else []
for f in (2, 3):
    for s in (1, 2):
        rows.append(run_fleet_pipeline(f, train_seed=s))
        json.dump(rows, open(out, "w"), indent=2)  # saved after each fleet/seed
