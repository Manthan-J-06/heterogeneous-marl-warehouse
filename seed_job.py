import json, os, argparse
import experiment_runner as er
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["blind", "aware"])
ap.add_argument("train_seed", type=int)
ap.add_argument("--fleet_seed", type=int, default=2)
ap.add_argument("--eval_only", action="store_true")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--config", default="configs/low_variance_fleet.yaml")
a = ap.parse_args()
if a.steps:
    er.TOTAL_STEPS = a.steps
er.CONFIG = a.config
stem = os.path.basename(a.config).replace(".yaml", "")
if a.eval_only:  # reuse an already-archived checkpoint
    from eval_policy import evaluate
    ck = os.path.join("archive", f"fleet{a.fleet_seed}", f"mappo_{a.mode}_{stem}_seed{a.train_seed}", "checkpoints")
    tot = 0
    for b in er.BASE_SEEDS:
        r = evaluate(config_path=er.CONFIG, fleet_seed=a.fleet_seed, episodes=er.EPISODES_PER_BASE,
                     base_seed=b, mappo_dir=ck)
        tot += r["total_deliveries"]
        print(f"base_seed={b}: {r['total_deliveries']}", flush=True)
else:
    tot = er.train_and_eval_mappo(a.fleet_seed, aware=(a.mode == "aware"), train_seed=a.train_seed)
os.makedirs("logs", exist_ok=True)
prefix = f"long{a.steps}" if a.steps else "res"
if stem != "low_variance_fleet":
    prefix += f"_{stem}"
out = f"logs/{prefix}_fleet{a.fleet_seed}_{a.mode}_seed{a.train_seed}.json"
json.dump({"fleet_seed": a.fleet_seed, "mode": a.mode, "train_seed": a.train_seed, "steps": a.steps,
           "config": a.config, "total": tot}, open(out, "w"))
print("saved", out, tot, flush=True)
