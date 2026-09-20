\# Experiment Configuration System



This document explains how to create and use experiment configuration files (YAML) to run the environment with different parameters — without touching any code.



\## How it works



All experiment settings live in `.yaml` files inside the `configs/` folder. The `config\_loader.py` script reads one of these files and passes the values into the environment automatically.



\## Schema



Every config file follows this structure:



```yaml

environment:

&#x20; grid\_size: 10          # Size of the grid (10 = 10x10 grid)

&#x20; num\_agents: 4           # Total number of agents in the environment

&#x20; episode\_length: 200     # Max number of steps per episode



heterogeneity:

&#x20; enabled: false          # Set to true in later phases to enable agent heterogeneity

&#x20; agent\_types: \[]         # e.g. \["scout", "carrier"] — used in later phases

```



\### Field descriptions



| Field | Type | Description |

|---|---|---|

| `environment.grid\_size` | int | Size of the (square) grid the agents operate on |

| `environment.num\_agents` | int | Total number of agents to initialize |

| `environment.episode\_length` | int | Maximum steps allowed before an episode truncates |

| `heterogeneity.enabled` | bool | Whether heterogeneous agent types are active (Phase 2+) |

| `heterogeneity.agent\_types` | list | List of agent type names, used once heterogeneity is enabled |



\## Example configs



Three example configs are provided in `configs/`:



\- `base\_config.yaml` — default/medium setup (10x10 grid, 4 agents)

\- `small\_grid.yaml` — smaller setup for quick tests (5x5 grid, 2 agents)

\- `large\_grid.yaml` — larger setup for scaling tests (20x20 grid, 8 agents)



\## Creating a new config



To create a new experiment configuration:



1\. Copy any existing file in `configs/` (e.g. `base\_config.yaml`)

2\. Rename it to describe your experiment (e.g. `my\_experiment.yaml`)

3\. Edit the values under `environment` as needed

4\. No code changes required!



\## Using a config in code



```python

from config\_loader import load\_config

from mock\_env import MockEnv



config = load\_config("configs/base\_config.yaml")

env = MockEnv(config=config)

```

## Heterogeneity Schema (Phase 2)

The `heterogeneity` block defines how agent parameters (speed, load capacity, battery) vary across the fleet. Two modes are supported:

```yaml
heterogeneity:
  enabled: true
  mode: uniform   # "uniform" or "explicit"

  # Used when mode is "uniform"
  uniform_ranges:
    speed: [0.5, 1.5]
    load_capacity: [1, 5]
    battery: [50, 100]

  # Used when mode is "explicit"
  explicit_fleet: []
```

### Field descriptions

| Field | Type | Description |
|---|---|---|
| `heterogeneity.enabled` | bool | Whether agents have varying parameters. If `false`, all agents use identical defaults. |
| `heterogeneity.mode` | string | `"uniform"` — agents randomly drawn from ranges. `"explicit"` — agents manually defined. |
| `heterogeneity.uniform_ranges.speed` | list | `[min, max]` speed range agents are randomly drawn from (uniform mode only). |
| `heterogeneity.uniform_ranges.load_capacity` | list | `[min, max]` load capacity range (uniform mode only). |
| `heterogeneity.uniform_ranges.battery` | list | `[min, max]` battery/energy range (uniform mode only). |
| `heterogeneity.explicit_fleet` | list | One entry per agent with exact `agent_id`, `speed`, `load_capacity`, `battery` values (explicit mode only). Length must equal `num_agents`. |

### Uniform mode example

Agents are randomly generated within the given ranges each run:

```yaml
uniform_ranges:
  speed: [0.5, 1.5]
  load_capacity: [1, 5]
  battery: [50, 100]
```

### Explicit mode example

Every agent's exact parameters are hand-defined:

```yaml
explicit_fleet:
  - agent_id: 0
    speed: 1.2
    load_capacity: 3
    battery: 90
  - agent_id: 1
    speed: 0.8
    load_capacity: 5
    battery: 60
```

### New example configs

- `low_variance_fleet.yaml` — narrow parameter ranges, agents are similar to each other
- `high_variance_fleet.yaml` — wide parameter ranges, agents are very different from each other
- `explicit_fleet_example.yaml` — demonstrates explicit mode with 3 hand-defined agents

### Generating a fleet in code

```python
from config_loader import load_config, build_fleet

config = load_config("configs/low_variance_fleet.yaml")
fleet = build_fleet(config)
# fleet is a list of dicts: [{"agent_id": 0, "speed": ..., "load_capacity": ..., "battery": ...}, ...]
```
