## PineappleRL Architecture – Current State (Brief)

### 1. Core Game Engine
- **C++ backend (`cpp/`, `ofc_cpp` module)**  
  - Implements fast OFC rules: deck handling, legal action generation, state transitions, scoring, and multi-env workers.  
  - Exposed to Python via pybind11 as `ofc_cpp` (e.g. `generate_random_episodes`, `create_engine`, `legal_actions_round0`, `legal_actions_rounds1to4`, `step_state_round0`, `step_state`, `score_board_from_ints`, `encode_state_batch_ints`).

- **Python wrapper (`ofc_env.py`)**  
  - `State`: 13-slot board (bottom/middle/top), round, current draw, deck, cards placed.  
  - `Action`: `(keep_indices, placements)` describing which cards to keep and where to place them.  
  - `OfcEnv`:
    - `reset()`: creates a shuffled deck, deals 5 cards, returns an empty board state.  
    - `legal_actions(state)`: enumerates legal moves, using C++ helpers when available and a Python fallback otherwise; includes a `_feasible` filter to remove obviously fouling placements.  
    - `step(state, action)`: applies an action using C++ (`step_state_round0` / `step_state`) or Python fallback.  
    - `score(state)`: returns the final OFC score (royalties – foul penalty) via C++ or Python scoring.
  - **Two-board helpers (for head-to-head):**
    - `reset_two_boards()`: returns two `State`s using the same shuffled deck.  
    - `step_two_boards(state_a, action_a, state_b, action_b)`: steps both boards and asserts decks remain in sync.

### 2. State Encoding & Network
- **`state_encoding.py`**  
  - Encodes a `State` into a fixed 838‑dim vector:  
    - 13×52 one-hot slots for board occupancy.  
    - Round one-hot, deck size, and current draw cards.  
  - `encode_state_batch` has a C++-accelerated path via `encode_state_batch_ints` and a Python/Numpy fallback.

- **`value_net.py`**  
  - `ValueNet`: legacy network used by `train.py`.  
    - Shared trunk -> multiple heads:  
      - `value_head`: predicts final score.  
      - `foul_head`: predicts foul probability.  
      - `round0_head`: auxiliary logits for round-0 placements.  
      - `feas_head`: feasibility-style auxiliary head.  
  - This net is trained via the current RL loop and via small supervised scripts.

### 3. Current RL Training (`train.py`)
- **SelfPlayTrainer** (single-player, score-focused):  
  - Uses `ofc_cpp.generate_random_episodes` to generate large numbers of random episodes and pushes encoded states + final scores into replay buffers.  
  - Uses a C++ “engine” (`create_engine`, `engine_start_envs`, `request_policy_batch`, `apply_policy_actions`, `engine_collect_encoded_episodes`) to generate model-guided episodes.  
  - Maintains separate buffers for random vs policy data and samples batches with score‑based weighting.

- **Training step (`train_step`)**  
  - Samples a batch from the buffers.  
  - Encodes states and computes:
    - Value loss (MSE on normalized final scores).  
    - Foul loss (BCE to classify fouls).  
    - Feasibility + round‑0 auxiliary losses.  
  - Combined loss heavily upweights fouls and large‑magnitude scores.  
  - Uses Adam, gradient clipping, and periodic checkpoints (`value_net_checkpoint_epXXXX.pth`).

- **Limitations observed in practice**  
  - Even with millions of hands, the resulting policy still fouls ≈50–60% of the time and rarely achieves royalty‑rich boards.  
  - Diagnostics show the network often predicts near-zero value and ~0.5 foul probability across many different boards—indicating that the current data mix and loss setup are not enough for it to learn strong structure.

### 4. Supervised Tools
- **`supervised_train.py`**  
  - Builds a synthetic dataset of boards labeled by their OFC scores using `score_board` / `score_board_from_ints`.  
  - Generates a mix of:
    - “good” boards (monotone/strong rows),  
    - “neutral” boards (random non-fouling layouts),  
    - “foul” boards (intentionally violating row ordering).  
  - Trains `ValueNet` as a pure regressor on these scores and writes:
    - `supervised_value_net.pth`  
    - `value_net_checkpoint_ep0.pth` (to seed RL).

- **`simple_train.py`**  
  - Earlier foul‑only pretraining: treats episodes as foul vs non‑foul and trains the foul head as a binary classifier.

### 5. Two-Player Simulation
- **`two_player_sim.py`**  
  - Uses `OfcEnv.reset_two_boards()` to give two players the same deck.  
  - Player A:
    - Either `ValueNetPolicy` (loads a `ValueNet` checkpoint and chooses actions via value–penalty*foul_prob)  
    - Or a random policy.  
  - Player B: random policy (separate RNG).  
  - Both play independently until their boards are complete, then are scored.  
  - Reports:
    - Wins / losses / ties.  
    - Foul rates for each player.  
    - Average score difference (A – B).
  - This is used as a head‑to‑head diagnostic: so far, the trained policy tends to tie random (mostly 0‑0, few royalties, frequent fouls), showing that it hasn’t yet learned a clearly superior strategy.

### 6. Overall Picture (Today)
- **Game logic & C++ backend**: Solid and fast; rules, state transitions, and scoring are well‑encapsulated in `ofc_cpp` and `OfcEnv`.
- **Current RL scheme**: Single‑player RL around `ValueNet` and C++ episode generators, with foul‑aware loss. In practice, it reduces fouls somewhat but converges to a policy that still fouls often and scores near zero.
- **Supervised & two‑player tools**:  
  - Supervised scripts create synthetic training data but haven’t yet produced a strongly discriminative value function.  
  - Two‑player sim is in place to evaluate learned policies vs baselines.

The next major step (your new plan) is to move toward an AlphaZero‑style policy+value network with MCTS‑based two‑player self‑play and a replay‑buffer trainer, implemented alongside the existing RL loop so you can experiment without losing the current pipeline.


