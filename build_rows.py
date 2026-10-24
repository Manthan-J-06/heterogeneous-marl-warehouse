import json, glob
ctrl = {r["Fleet"]: r for r in json.load(open("logs/seed42_rows.json"))}
res = {}
for p in glob.glob("logs/res_fleet*_seed*.json"):
    d = json.load(open(p))
    res.setdefault((d["fleet_seed"], d["train_seed"]), {})[d["mode"]] = d["total"]
rows = []
for (f, s), m in sorted(res.items()):
    if "blind" in m and "aware" in m:
        c = ctrl[f"Fleet {f}"]
        rows.append({"Fleet": f"Fleet {f} (seed {s})", "Commit": "local", "random": c["random"],
                     "blind": m["blind"], "aware": m["aware"], "fwd40": c["fwd40"], "left40": c["left40"]})
json.dump(rows, open("logs/seed_rows.json", "w"), indent=2)
print(json.dumps(rows, indent=1))
