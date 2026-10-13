"""
Extended evaluation runner for Option C:
1. High Variance Fleet (configs/high_variance_fleet.yaml) evaluation across controls, blind, and aware.
2. Architecture / Information-control ablation: Aware architecture with Zeroed / Shuffled fleet features.
   This has the exact same input dimensions as 'aware', testing whether the slight margin over blind
   is merely an extra-parameter/initialization artifact or genuine information usage.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import torch
import numpy as np

PYTHON = sys.executable
BASE_SEEDS = [1000, 2000, 3000, 4000]
EPISODES_PER_BASE = 50

def run_high_variance_experiment(fleet_seed=1, train_seed=42):
    print(f"\n=======================================================")
    print(f"  PART 1: HIGH VARIANCE FLEET (fleet_seed={fleet_seed})")
    print(f"=======================================================")
    config_path = "configs/high_variance_fleet.yaml"
    
    # 1. Controls
    from eval_controls import run_controls
    print(f"[Controls] Running eval_controls for High Variance Fleet (fleet_seed={fleet_seed})...")
    env, res_ctrl = run_controls(config_path, fleet_seed, base_seeds=BASE_SEEDS, episodes=EPISODES_PER_BASE)
    print("Controls results:", res_ctrl)
    
    # 2. Train Blind & Aware (30k steps)
    from experiment_runner import run_cmd
    modes = [("blind", False), ("aware", True)]
    deliv_results = {}
    
    for mode_str, is_aware in modes:
        print(f"\n[Training] MAPPO {mode_str} on High Variance Fleet...")
        cmd = [
            PYTHON, "train_mappo.py",
            "--config", config_path,
            "--fleet_seed", str(fleet_seed),
            "--total-steps", "30000",
            "--seed", str(train_seed)
        ]
        if is_aware:
            cmd.append("--aware")
            
        log_path = f"logs/train_highvar_{mode_str}_fleet{fleet_seed}_seed{train_seed}.log"
        t0 = time.time()
        run_cmd(cmd, log_file=log_path)
        print(f"Training completed in {time.time()-t0:.1f}s.")
        
        # Verify and Archive
        run_name = f"mappo_{mode_str}_high_variance_fleet_seed{train_seed}"
        src_dir = os.path.join("runs", run_name)
        archive_dir = os.path.join("archive", f"fleet{fleet_seed}_highvar", run_name)
        os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
        if os.path.exists(archive_dir):
            shutil.rmtree(archive_dir)
        shutil.move(src_dir, archive_dir)
        
        # Evaluate
        from eval_policy import evaluate
        ckpt_dir = os.path.join(archive_dir, "checkpoints")
        tot = 0
        for b in BASE_SEEDS:
            r = evaluate(config_path, fleet_seed, EPISODES_PER_BASE, base_seed=b, mappo_dir=ckpt_dir)
            tot += r["total_deliveries"]
        deliv_results[mode_str] = tot
        print(f"Evaluation {mode_str}: {tot} deliveries over 200 eps")

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    return {
        "Config": "High Variance Fleet 1",
        "Commit": commit,
        "random": res_ctrl.get("uniform", 0),
        "blind": deliv_results["blind"],
        "aware": deliv_results["aware"],
        "fwd40": res_ctrl.get("fwd40", 0),
        "left40": res_ctrl.get("left40", 0)
    }

def run_zeroed_features_ablation(fleet_seed=2, train_seed=42):
    print(f"\n=======================================================")
    print(f"  PART 2: ZEROED-FEATURES CONTROL ABLATION (Fleet {fleet_seed})")
    print(f"=======================================================")
    # Evaluates the checkpoint trained with aware architecture on fleet 2,
    # but feeds zeroed feature inputs at inference, testing if performance changes.
    from config_loader import load_config, build_fleet
    from launch_experiment import fleet_to_agents_cfg
    import heterogeneous_env as he
    from algorithms.mappo import MAPPOTrainer
    import random

    config_path = "configs/low_variance_fleet.yaml"
    c = load_config(config_path)
    random.seed(fleet_seed)
    fleet = build_fleet(c)
    c["heterogeneity"]["agents"] = fleet_to_agents_cfg(fleet)
    env = he.build_heterogeneous_env(c)
    n = len(env.action_space)

    ckpt_dir = os.path.join("archive", f"fleet{fleet_seed}", f"mappo_aware_low_variance_fleet_seed{train_seed}", "checkpoints")
    meta = json.load(open(os.path.join(ckpt_dir, "fleet.json")))
    
    cfg = dict(c)
    cfg.update(gamma=.99, gae_lambda=.95, clip_eps=.2, entropy_coef=.01, value_loss_coef=.5, ppo_epochs=1)
    
    # Standard aware props
    aware_props = [[sp, cp / 10.0, bt / 1000.0] for sp, cp, bt in zip(env.speeds, env.capacities, env.battery_capacities)]
    # Zeroed props (same dimensions, zero info)
    zero_props = [[0.0, 0.0, 0.0] for _ in range(n)]

    ck = torch.load(os.path.join(ckpt_dir, "mappo_final.pt"), map_location="cpu", weights_only=False)

    def eval_with_props(props, label):
        trainer = MAPPOTrainer(env.obs_dim, env.obs_dim * n, n, env.n_actions, cfg, agent_props=props)
        trainer.actor.load_state_dict(ck["actor"])
        trainer.critic.load_state_dict(ck["critic"])
        total_d = 0
        for b in BASE_SEEDS:
            for ep in range(EPISODES_PER_BASE):
                s = b + ep
                random.seed(s)
                np.random.seed(s)
                obs, _ = env.reset(seed=s)
                for t in range(500):
                    actions, _, _ = trainer.act(obs, env.global_state(obs))
                    obs, r, term, trunc, info = env.step(actions)
                    total_d += info["deliveries"]
                    if term or trunc:
                        break
        print(f"[{label}] Deliveries over 200 eps: {total_d}")
        return total_d

    standard_deliv = eval_with_props(aware_props, "Actual Fleet Features")
    zero_deliv = eval_with_props(zero_props, "Zeroed Features Control")
    return standard_deliv, zero_deliv

if __name__ == "__main__":
    r_highvar = run_high_variance_experiment(fleet_seed=1, train_seed=42)
    std_d, zero_d = run_zeroed_features_ablation(fleet_seed=2, train_seed=42)
    
    print("\n" + "="*50)
    print("A3 SUMMARY TABLE")
    print("="*50)
    print(f"| {r_highvar['Config']} | {r_highvar['Commit']} | random: {r_highvar['random']} | blind: {r_highvar['blind']} | aware: {r_highvar['aware']} | fwd40: {r_highvar['fwd40']} | left40: {r_highvar['left40']} |")
    print(f"\nZeroed Features Control Ablation (Fleet 2 Aware Model):")
    print(f"Actual Fleet Features: {std_d} deliveries")
    print(f"Zeroed Fleet Features: {zero_d} deliveries")
