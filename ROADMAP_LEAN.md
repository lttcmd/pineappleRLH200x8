## Lean OFC Roadmap

### 1. Baseline SFL (Done)
- C++ port of xeond8 “Simple Fantasy Like” logic (`cpp/sfl_policy.cpp`) with deterministic Monte Carlo completions.
- Pybind helpers `sfl_choose_action` / `generate_sfl_dataset` feed Python without extra glue.
- Unit test `tests/test_sfl_policy.py` verifies parity against a reference Python translation.

### 2. PolicyNet Supervised Loop (Ready)
- `collect_sfl_dataset_cpp.py` produces shards with state encodings, per-action next states, and labels.
- `train_policy_net.py` streams shards via `ShardedPolicyIterableDataset`, supports pin-memory, AMP, grad accumulation, and reports true sample progress.
- Goal: pretrain PolicyNet to imitate SFL’s street-wise choices before touching RL.

### 3. Additional Heuristic/Search Bots
| Bot | Source | Status | Integration Plan |
| --- | --- | --- | --- |
| ISMCTS | xeond8 `bot/ismcts.py` | Pending | Port core tree logic to C++ (`cpp/ismcts.cpp`), reuse SFL for rollouts, expose via pybind. |
| MCTS (one-board / two-board) | xeond8 `bot/mcts.py` | Optional | Similar approach: C++ search loop + SFL simulations. |
| SimpleFantasyTwoBoards | xeond8 | Optional later | Could reuse same SFL primitives once head-to-head support needed. |

### 4. Evaluation Harness
- Add simple CLI to pit any C++ bot or trained PolicyNet against the native scoring engine (single-player foul rate + royalties, later head-to-head when ISMCTS arrives).
- Targets: `policy_net_policy.py` wrapper plus a minimal evaluator script (replacement for old `two_player_sim.py`).

### 5. RL / Fine-Tuning
- Start from the supervised PolicyNet weights.
- Use the existing pybind engine for fast rollouts; keep the python side to batching + optimizer steps.
- Regularize toward the supervised policy (KL or L2 on logits) to prevent catastrophic foul rate spikes.

### 6. Stretch Goals
- Dataset variants (ISMCTS, hybrid heuristics).
- Curriculum schedules: start with SFL imitation, mix in ISMCTS labels, then RL.
- Tournament tooling: scheduled head-to-head runs among bots + learned policies.

With SFL + PolicyNet training already in place, the next concrete tasks are:
1. Port ISMCTS into the same C++ module.
2. Build the lightweight evaluator to compare SFL vs PolicyNet vs ISMCTS.
3. Kick off the first supervised training run using the refreshed harness.***

