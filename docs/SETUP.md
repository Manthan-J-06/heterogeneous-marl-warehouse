# Environment Setup & Validation Guide

This document describes how to set up, reproduce, and validate the base multi-agent simulation environment for the heterogeneous MARL warehouse project.

---

## 1. Prerequisites

- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python Version:** Python 3.10 – 3.14 (Validated on Python 3.14)
- **Git:** Installed and available on system `PATH`

---

## 2. Environment Installation Steps

### Step 2.1: Clone the Repository & Checkout Branch
```bash
git clone <repository-url>
cd heterogeneous-marl-warehouse
git checkout feature/base-env-setup
```

### Step 2.2: Create and Activate Virtual Environment
```bash
# Windows (Command Prompt / PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2.3: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. Key Dependencies & Versions

| Package | Version | Purpose |
|---|---|---|
| `rware` | `2.0.0` | Multi-Robot Warehouse simulation environment |
| `gymnasium` | `1.3.0` | Farama Gymnasium standard environment API |
| `pettingzoo` | `1.27.0` | Multi-Agent Reinforcement Learning API |
| `torch` | `2.14.0` | PyTorch deep learning framework |
| `numpy` | `2.5.3` | Numerical computing and array manipulation |
| `pyyaml` | `6.0.3` | YAML configuration parsing |
| `matplotlib` | `3.11.2` | Evaluation metrics plotting and visualization |
| `pandas` | `3.0.6` | Tabular metrics and experiment data processing |
| `tensorboard` | `2.21.0` | Training curves and real-time logging |

---

## 4. Environment Configuration

The initial test configuration is stored in [`configs/env_small.yaml`](../configs/env_small.yaml):

```yaml
environment:
  # RWARE environment ID. "rware-tiny-2ag-v2" is the smallest built-in config:
  # ~10x11 grid, 2 agents, 1 request queue size.
  env_id: "rware-tiny-2ag-v2"

validation:
  # Number of episodes to run for the random-policy validation.
  num_episodes: 20
  # Random seed for reproducibility.
  seed: 42
```

### Environment Specs (`rware-tiny-2ag-v2`)
- **Agents:** 2
- **Grid Layout:** Tiny (~10 × 11 grid layout with shelf racks and delivery stations)
- **Action Space (per agent):** `Discrete(5)`
  - `0`: Turn Left
  - `1`: Turn Right
  - `2`: Move Forward
  - `3`: Load / Unload Shelf
  - `4`: No-op (Wait)
- **Observation Space (per agent):** `Box(low=-inf, high=inf, shape=(115,), dtype=float32)` (or discrete grid features)
- **Max Steps:** 500 steps per episode

---

## 5. Running the Validation Script

To verify that the simulation environment resets, steps, and terminates properly with no exceptions or deadlocks, run:

```bash
python scripts/validate_env.py --config configs/env_small.yaml
```

### Expected Output
The script executes 20 random-policy rollout episodes and prints a summary:

```text
======================================================================
  PHASE 1 — RWARE ENVIRONMENT VALIDATION
  Random-policy rollout test
======================================================================

Creating environment: rware-tiny-2ag-v2 ...
======================================================================
  Environment: rware-tiny-2ag-v2
  Number of agents: 2
  Observation space (per agent): Box(-inf, inf, (115,), float32)
  Action space (per agent): Discrete(5)
  Max steps (from env): 500
======================================================================

  Episode   1/20 — steps:  500, reward:     0.00, TERMINATED
  Episode   2/20 — steps:  500, reward:     0.00, TERMINATED
  ...
  Episode  20/20 — steps:  500, reward:     0.00, TERMINATED

======================================================================
  VALIDATION SUMMARY
======================================================================
Episode    | Steps    | Total Reward   | Outcome     
------------------------------------------------------
1          | 500      | 0.00           | TERMINATED  
2          | 500      | 0.00           | TERMINATED  
...
20         | 500      | 0.00           | TERMINATED  
------------------------------------------------------
TOTAL      | 10000    | 0.00           |
MEAN       | 500.0    | 0.00           |

  All 20 episodes completed successfully.
  Environment resets, steps, and terminates correctly.
======================================================================
  Total wall time: ~0.85s
```

---

## 6. Technical Notes & Gotchas

1. **RWARE Environment Termination vs Truncation:**
   - In RWARE `2.0.0` with Gymnasium `1.3.0`, episodes reach their step limit (`500` steps) by returning `terminated = True` (rather than `truncated = True`). The validation script handles both flags gracefully.
2. **Reward with Random Actions:**
   - A total reward of `0.00` under a purely random policy is expected. In warehouse coordination, delivering a requested shelf requires a coordinated sequence of picking up a specific shelf, navigating to a goal station, and unloading before the request expires.
3. **Multi-Agent Step Return Format:**
   - `env.step(actions)` returns `obs` as a tuple of `n_agents` arrays, `rewards` as a list/tuple of floats per agent, `terminated` and `truncated` booleans, and an `info` dictionary.
