# Experiment Configuration System

This document explains how to create and use experiment configuration files (YAML) to run the environment with different parameters — without touching any code.

## How it works

All experiment settings live in `.yaml` files inside the `configs/` folder. The `config_loader.py` script reads one of these files and passes the values into the environment automatically. `launch_experiment.py` uses the same config to build and run the real environment end-to-end.

## Schema

Every config file follows this structure:

```yaml
environment:
  grid_size: 10                  # Size of the grid (10 = 10x10 grid)
  num_agents: 4                  # Total number of agents in the environment
  episode_length: 200            # Max number of steps per episode
  env_id: rware-tiny-4ag-v2      # Registered RWARE environment name (must match num_agents)

heterogeneity:
  enabled: true
  mode: uniform                  # "uniform" or "explicit"
  energy_weight: 0.0             # How much battery usage penalizes reward

  # Used when mode is "uniform" — agents randomly drawn from these ranges
  # NOTE: speed must be between 0.0 and 1.0
  uniform_ranges:
    speed: [0.6, 1.0]
    load_capacity: [1, 5]
    battery: [50, 100]

  # Used when mode is "explicit" — ignored unless mode is "explicit"
  explicit_fleet: []
```

### Field descriptions

| Field | Type | Description |
|---|---|---|
| `environment.grid_size` | int | Size of the (square) grid the agents operate on |
| `environment.num_agents` | int | Total number of agents to initialize |
| `environment.episode_length` | int | Maximum steps allowed before an episode truncates |
| `environment.env_id` | string | Registered RWARE environment name. The agent count in the name must match `num_agents` |
| `heterogeneity.enabled` | bool | Whether agents have varying parameters. If `false`, all agents use identical defaults |
| `heterogeneity.mode` | string | `"uniform"` — agents randomly drawn from ranges. `"explicit"` — agents manually defined |
| `heterogeneity.energy_weight` | float | How much battery/energy usage penalizes each agent's reward |
| `heterogeneity.uniform_ranges.speed` | list | `[min, max]` speed range, must stay within 0.0–1.0 (uniform mode only) |
| `heterogeneity.uniform_ranges.load_capacity` | list | `[min, max]` load capacity range (uniform mode only) |
| `heterogeneity.uniform_ranges.battery` | list | `[min, max]` battery/energy range (uniform mode only) |
| `heterogeneity.explicit_fleet` | list | One entry per agent with exact `agent_id`, `speed`, `load_capacity`, `battery` values (explicit mode only). Length must equal `num_agents` |

### Uniform mode example

Agents are randomly generated within the given ranges each run:

```yaml
uniform_ranges:
  speed: [0.6, 1.0]
  load_capacity: [1, 5]
  battery: [50, 100]
```

### Explicit mode example

Every agent's exact parameters are hand-defined:

```yaml
explicit_fleet:
  - agent_id: 0
    speed: 0.9
    load_capacity: 3
    battery: 90
  - agent_id: 1
    speed: 0.7
    load_capacity: 5
    battery: 60
```

## Example configs

Example configs are provided in `configs/`:

- `base_config.yaml` — default setup (10x10 grid, 4 agents, uniform heterogeneity)
- `small_grid.yaml` — smaller setup for quick tests
- `large_grid.yaml` — larger setup for scaling tests
- `low_variance_fleet.yaml` — narrow parameter ranges, agents are similar to each other
- `high_variance_fleet.yaml` — wide parameter ranges, agents are very different from each other
- `explicit_fleet_example.yaml` — demonstrates explicit mode with 3 hand-defined agents

## Creating a new config

To create a new experiment configuration:

1. Copy any existing file in `configs/` (e.g. `base_config.yaml`)
2. Rename it to describe your experiment (e.g. `my_experiment.yaml`)
3. Edit the values as needed
4. No code changes required!

## Generating a fleet in code

```python
from config_loader import load_config, build_fleet

config = load_config("configs/low_variance_fleet.yaml")
fleet = build_fleet(config)
# fleet is a list of dicts: [{"agent_id": 0, "speed": ..., "load_capacity": ..., "battery": ...}, ...]
```

## Experiment Launcher

`launch_experiment.py` is a config-driven launcher — given a config file, it builds the fleet, initializes the real heterogeneous environment, and runs a short test episode.

### Usage


## Ablation Runner

`run_ablation.py` runs multiple configs automatically and combines their results into one comparison table — useful for comparing different heterogeneity levels, grid sizes, or fleet setups side by side.

### Usage

Pass any number of config file paths, separated by spaces. Each one runs for 5 episodes by default.

### What it does

1. For each config: loads it, builds the fleet, builds the real environment, and runs 5 episodes using Manthan's `MetricsLogger`
2. Computes summary stats per config: average throughput (tasks/episode), average energy per completed task, average episode length, and per-agent average reward (workload)
3. Prints a comparison table to the console
4. Saves all results combined into `ablation_results.csv` — one row per config, easy to compare in a spreadsheet

### Interpreting results

- **Avg Throughput**: average tasks completed per episode. Higher is better.
- **Avg Energy Per Task**: total energy spent divided by tasks completed. Lower is better. Shows as `N/A` if zero tasks were completed (can't divide by zero).
- **Avg Episode Length**: average number of steps before an episode ends (either by completion or truncation).
- **Per-Agent Avg Reward**: shows workload distribution — whether some agents are contributing much more/less than others.

### Important note on random actions

By default, `run_ablation.py` uses **random actions** for each agent (no trained policy yet). This means task completion is rare, so `Avg Throughput` and `Avg Energy Per Task` will often show `0.0` / `N/A` — this is expected and not a bug. Warehouse tasks (navigate → pick up → carry → drop) require a specific sequence of correct actions, which random movement rarely produces by chance.

The ablation runner's current purpose is to validate the **infrastructure**: config loading → environment setup → metric logging → comparison across runs. Once trained policies (MAPPO/QMIX, per the project's Phase 2 plan) are plugged in in place of random actions, throughput and energy metrics will become meaningful for comparing heterogeneity 
levels.