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

