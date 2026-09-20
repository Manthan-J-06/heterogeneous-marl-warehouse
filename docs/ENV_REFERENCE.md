# Environment Reference: RWARE and TA-RWARE

> **Which environment we actually use:** this project uses plain RWARE
> (`rware-tiny-2ag-v2`), not TA-RWARE. Section 1's TA-RWARE material below is
> background/comparison only. The relevant sections for our actual setup are
> the "Original RWARE" subsections in Sections 2-4 (5 discrete actions:
> Noop/Forward/Left/Right/Load-Unload).

Internal reference for the base environment used in this project. Written for teammates who have never seen RWARE.

**Sources**
- TA-RWARE (task-assignment variant, the one closest to our heterogeneous fleet): https://github.com/uoe-agents/task-assignment-robotic-warehouse
- Original RWARE: https://github.com/uoe-agents/robotic-warehouse
- Paper describing TA-RWARE: "Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers" (arXiv 2212.11498)

> **Verification status.** This doc was written from the official READMEs and the paper. The source files (`tarware/warehouse.py` etc.) could not be opened while writing, so anything marked **[VERIFY]** is an inference that someone should confirm by running the environment or reading the code. Section 7 lists the exact checks.

---

## 1. What RWARE is (30-second version)

A simulated warehouse on a grid. Robots move around, pick up shelves, carry them to a goal (workstation) where a human picks the items, and then return each shelf to an empty shelf location. At any moment a fixed number of shelves, R, are "requested". When a requested shelf is delivered, a new random shelf becomes requested.

There are two related environments:

| | Original RWARE | TA-RWARE |
|---|---|---|
| Agent types | All identical | **Two types: AGVs (carriers) and Pickers (loaders)** |
| What an action means | Low-level move (turn, forward, load/unload) | **A target location** on the map |
| Path-finding | Agent does it step by step | Built in (A*) |
| Observation | Small window around the agent | Wide view of the whole warehouse (global or partial) |

Our project is about heterogeneous fleets, so **TA-RWARE is the relevant base**. The original RWARE is described only for comparison.

---

## 2. Observation space

### TA-RWARE

Each agent gets its own observation every timestep. The environment exposes `env.observation_space` as a tuple with one entry per agent, so `len(env.observation_space) == env.n_agents`, where `env.n_agents = env.n_agvs + env.n_pickers`.

There are two modes, chosen when the environment is created (the "observability type": `partial` or `global`).

**Global mode.** Every agent sees the same information:
- The current target and location of every agent.
- For AGVs: whether each is carrying a shelf, and whether the carried shelf is a requested one.
- For AGVs: their loading status.
- For every shelf location: whether it is occupied and whether it is requested.

**Partial mode.** The same information with some parts removed per agent type:
- AGVs cannot see the carrying/loading status of *other* AGVs.
- Pickers cannot see *any* shelf information.

**Structure and dimensions: [VERIFY].** The README does not give the array shape. Because agents see the whole warehouse, the size depends on the number of agents and the number of shelf locations, so it grows with the map. Get the real numbers with `env.observation_space` (see Section 7). Do not hard-code a size in our code.

### Original RWARE (for comparison)

Partially observable: a small square window centred on the agent, 3x3 by default (configurable). Inside it the agent sees its own position, rotation and whether it carries a shelf, other robots' position and rotation, and shelves along with whether each is currently requested. Observation is a flat vector (`Box`) per agent.

---

## 3. Action space

### TA-RWARE

Actions are **discrete and location-based**. An action is an index that means "go to this location".

| Agent type | Available actions |
|---|---|
| AGV | Shelf locations + Goal locations |
| Picker | Shelf locations only (goal locations are masked out, since Pickers do not deliver) |

- The agent chooses *where* to go, not how to get there. Path-finding is done by A*.
- Collisions: if agent i steps onto a current or future position of agent j, agent i enters a `fixing_clash` state and recomputes its path around the other agents. This can deadlock, so agents get a fixed window of timesteps to re-plan. If no path is found in time, they become available again and can pick a new target.
- The design forces AGVs and Pickers to coordinate: they need to meet at the same shelf location at the same time for a pick.
- **The action space is large** and scales with the warehouse layout. This is called out as one of the main challenges of the environment.
- Exact size and whether there is an extra no-op action: **[VERIFY]**. The README example prints sampled actions in the 100+ range on the tiny map, so expect a few hundred options on larger maps.

The environment also ships a hand-written FIFO baseline heuristic (`tarware/heuristic.py`, runnable via `scripts/run_heuristic.py`). It assigns the closest free AGV and Picker to the first request in the queue. This is useful as a performance baseline.

### Original RWARE

Five discrete actions per agent: `Noop`, `Forward`, `Left`, `Right`, `Toggle Load/Unload shelf` (the README shows `Tuple(Discrete(5), Discrete(5))` for 2 agents).

---

## 4. Default reward structure

### TA-RWARE

| Agent | Reward | When |
|---|---|---|
| AGV | **+1** | It delivers a requested shelf to a goal location |
| Picker | **+0.1** | It helps an AGV load or unload a shelf |
| All (observed in README example) | **-0.001** | Appears on each step in the README's example output **[VERIFY]** |

Key facts:
- Rewards are returned as a **list with one value per agent**, not one shared number. `terminated` and `truncated` are also per-agent lists (unlike standard Gymnasium).
- **Sparse reward.** An AGV has to travel to a shelf, load it, deliver it, and then find an empty location to return the previous shelf. Many steps pass between rewards. The README highlights this as a significant challenge.
- The README's description text does not mention the -0.001 value. It only shows up in the example printout, so check the code before relying on it.
- **Order of return values.** The README writes the step call as `n_obs, reward, truncated, terminated, info = env.step(actions)`. Standard Gymnasium order is `terminated, truncated`. **[VERIFY]** which order the code actually uses before writing any training loop.

### Original RWARE

Agents are rewarded (1 point) for delivering a requested shelf to a goal location. It is a sparse signal because only deliveries are rewarded. Reward can be set to cooperative or individual.

---

## 5. What is configurable without touching RWARE's code

These are exposed as environment parameters, or through registered environment names in `gym.make(...)`.

| Setting | Notes |
|---|---|
| Warehouse size / layout | Set by number of rack rows, rack columns, and shelves per rack. The README also lists a "custom layout" option (details not in the parts reviewed **[VERIFY]**). |
| Number of agents | Total count. |
| AGV : Picker ratio | Set the number of AGVs and Pickers separately. |
| Number of requested shelves R | Changes difficulty. Small R makes rewards sparser. |
| Observability | `partial` or `global`. |
| Preset environments | Names look like `tarware-tiny-3agvs-2pickers-partialobs-v1`, covering size, AGV count, Picker count and observability. |

Example from the README:

```python
import tarware
import gymnasium as gym
env = gym.make("tarware-tiny-3agvs-2pickers-partialobs-v1")
```

---

## 6. What likely needs deeper modification

Everything below is inferred from how the environment is described. Confirm in the source before planning work around it **[VERIFY]**.

| Change | Why it likely needs code changes |
|---|---|
| **Reward values** (1 / 0.1 / step penalty) | Described as fixed behaviour of the environment. No reward parameters are listed in the README. They are probably hard-coded in the step logic. Easy to edit, but it *is* a code change. |
| **Reward shaping** (e.g. distance bonuses, penalties for idle time) | Same reason. Alternatively this could be done in a **wrapper** around the environment without editing the core code. |
| **New agent types beyond AGV and Picker** | The two-type split (with its action masks and observation differences) appears to be built in. |
| **Per-agent properties for heterogeneity** (speed, capacity, battery, different sensors) | No such parameters are listed. Likely requires changes to the agent class and step logic. |
| **New observation content or a different partial-view rule** | Only two observation modes exist. Anything else means editing the observation-building code (or wrapping the env). |
| **Different action semantics** (e.g. going back to low-level moves) | Location-based actions and A* traversal are core design decisions. |
| **Collision / clash handling, deadlock window** | Part of the movement logic. |

**Practical note for the team:** anything that only *transforms* observations, rewards or actions (normalisation, reward shaping, action masking, frame-stacking) can usually be done in a Gymnasium **wrapper** without forking the environment. Anything that changes *how the world works* (new agent abilities, new dynamics) needs a change inside the environment.

---

## 7. Checks to close the [VERIFY] items

Once the environment is installed (`pip install -e .` from the TA-RWARE repo), run:

```python
import tarware
import gymnasium as gym

env = gym.make("tarware-tiny-3agvs-2pickers-partialobs-v1")
print(env.n_agents, env.n_agvs, env.n_pickers)
print(env.observation_space)   # per-agent shapes -> fills in Section 2
print(env.action_space)        # per-agent sizes  -> fills in Section 3

obs, info = env.reset(seed=0)  # reset may return only obs in this version, check
out = env.step(env.action_space.sample())
print(out)                     # check the order of terminated/truncated and the per-step reward
```

Also open these files and answer the open questions:
- `tarware/warehouse.py`: reward values, step penalty, constructor arguments, observation building.
- `tarware/heuristic.py`: how the baseline picks targets.
- `tarware/__init__.py`: the registered environment names.

Update this doc with the confirmed values and remove the **[VERIFY]** tags as they are settled.
