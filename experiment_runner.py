"""
Automated experiment runner for MARL warehouse.
Handles:
1. Running baseline controls (uniform/random, fwd40, left40) over 4 base seeds (1000, 2000, 3000, 4000) x 50 eps = 200 eps.
2. Training MAPPO (blind and aware) with verification:
   - fleet.json matches fleet seed
   - mappo_final.pt is newer than fleet.json
   - Moving run folder to archive to avoid stale/overwritten files
3. Evaluating MAPPO checkpoints across the 4 base seeds (1000, 2000, 3000, 4000) x 50 eps = 200 eps.
4. Outputting a markdown results table row per fleet.
"""

import os
import sys
import json
import time
import shutil
import subprocess

PYTHON = sys.executable
CONFIG = "configs/low_variance_fleet.yaml"
TOTAL_STEPS = 30000
BASE_SEEDS = [1000, 2000, 3000, 4000]
EPISODES_PER_BASE = 50

def run_cmd(cmd_list, env=None, log_file=None):
    print(f"--> Executing: {' '.join(cmd_list)}")
    if log_file:
        with open(log_file, "w") as f:
            proc = subprocess.run(cmd_list, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd_list, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        if not log_file:
            print("ERROR Output:", proc.stdout, proc.stderr)
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd_list)}")
    return proc

def get_controls(fleet_seed):
    print(f"\n[Controls] Running eval_controls for fleet_seed={fleet_seed}...")
    from eval_controls import run_controls
    env, res = run_controls(CONFIG, fleet_seed, base_seeds=BASE_SEEDS, episodes=EPISODES_PER_BASE)
    # res has: 'uniform', 'fwd40', 'left40'
    print(f"Controls results: uniform={res.get('uniform')}, fwd40={res.get('fwd40')}, left40={res.get('left40')}")
    return res

def train_and_eval_mappo(fleet_seed, aware=False, train_seed=42):
    mode_str = "aware" if aware else "blind"
    print(f"\n[Training] Starting MAPPO {mode_str} for fleet_seed={fleet_seed}, train_seed={train_seed}...")
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    log_path = f"logs/train_mappo_{mode_str}_fleet{fleet_seed}_seed{train_seed}.log"
    
    cmd = [
        PYTHON, "train_mappo.py",
        "--config", CONFIG,
        "--fleet_seed", str(fleet_seed),
        "--total-steps", str(TOTAL_STEPS),
        "--seed", str(train_seed)
    ]
    if aware:
        cmd.append("--aware")
        
    t0 = time.time()
    run_cmd(cmd, log_file=log_path)
    elapsed = time.time() - t0
    print(f"[Training] {mode_str} completed in {elapsed:.1f}s (~{elapsed/60:.2f}m).")
    
    # Identify run directory in runs/
    # train_mappo creates: runs/mappo_<aware|blind>_<config_name>_seed<seed>
    config_name = os.path.basename(CONFIG).replace(".yaml", "")
    run_name = f"mappo_{mode_str}_{config_name}_seed{train_seed}"
    src_dir = os.path.join("runs", run_name)
    ckpt_dir = os.path.join(src_dir, "checkpoints")
    fleet_json = os.path.join(ckpt_dir, "fleet.json")
    final_pt = os.path.join(ckpt_dir, "mappo_final.pt")
    
    # Verification
    if not os.path.exists(fleet_json):
        raise FileNotFoundError(f"Missing {fleet_json}")
    if not os.path.exists(final_pt):
        raise FileNotFoundError(f"Missing {final_pt}")
        
    with open(fleet_json) as f:
        meta = json.load(f)
    if meta.get("fleet_seed") != fleet_seed:
        raise ValueError(f"Fleet seed mismatch in fleet.json! Expected {fleet_seed}, got {meta.get('fleet_seed')}")
    if meta.get("aware") != aware:
        raise ValueError(f"Aware flag mismatch in fleet.json! Expected {aware}, got {meta.get('aware')}")
        
    t_fleet = os.path.getmtime(fleet_json)
    t_pt = os.path.getmtime(final_pt)
    if t_pt < t_fleet:
        raise ValueError(f"Stale checkpoint! {final_pt} is older than {fleet_json}")
        
    print(f"[Verification Passed] fleet.json matches fleet={fleet_seed}, aware={aware}. mappo_final.pt is fresh.")
    
    # Move to archive/fleet<N>/run_name
    archive_fleet_dir = os.path.join("archive", f"fleet{fleet_seed}")
    os.makedirs(archive_fleet_dir, exist_ok=True)
    dst_dir = os.path.join(archive_fleet_dir, run_name)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.move(src_dir, dst_dir)
    print(f"[Archived] Moved {src_dir} -> {dst_dir}")
    
    # Evaluate across 4 base seeds
    archived_ckpt_dir = os.path.join(dst_dir, "checkpoints")
    total_deliveries = 0
    from eval_policy import evaluate
    for b in BASE_SEEDS:
        res = evaluate(
            config_path=CONFIG,
            fleet_seed=fleet_seed,
            episodes=EPISODES_PER_BASE,
            base_seed=b,
            mappo_dir=archived_ckpt_dir
        )
        total_deliveries += res["total_deliveries"]
        print(f"  base_seed={b}: deliveries={res['total_deliveries']}/{EPISODES_PER_BASE} eps")
        
    print(f"[Evaluation] {mode_str} total deliveries over 200 eps: {total_deliveries}")
    return total_deliveries

def run_fleet_pipeline(fleet_seed, train_seed=42):
    print(f"\n=======================================================")
    print(f"  PIPELINE FOR FLEET {fleet_seed} (train_seed={train_seed})")
    print(f"=======================================================")
    controls = get_controls(fleet_seed)
    blind_deliv = train_and_eval_mappo(fleet_seed, aware=False, train_seed=train_seed)
    aware_deliv = train_and_eval_mappo(fleet_seed, aware=True, train_seed=train_seed)
    
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    row = {
        "Fleet": f"Fleet {fleet_seed} (seed {train_seed})" if train_seed != 42 else f"Fleet {fleet_seed}",
        "Commit": commit,
        "random": controls.get("uniform", 0),
        "blind": blind_deliv,
        "aware": aware_deliv,
        "fwd40": controls.get("fwd40", 0),
        "left40": controls.get("left40", 0)
    }
    print(f"\nResult Row: | {row['Fleet']} | {row['Commit']} | {row['random']} | {row['blind']} | {row['aware']} | {row['fwd40']} | {row['left40']} |")
    return row

if __name__ == "__main__":
    if len(sys.argv) > 1:
        f_seed = int(sys.argv[1])
        t_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        run_fleet_pipeline(f_seed, t_seed)
    else:
        # Default run test on fleet 2
        run_fleet_pipeline(2)
