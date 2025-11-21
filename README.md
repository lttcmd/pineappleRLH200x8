# Pineapple RL Optimization

A high-performance Open Face Chinese Poker (OFC) reinforcement learning system with C++ game engine and Python RL training.

## Architecture

### Core Components

1. **C++ Game Engine** (`cpp/`)
   - High-performance game logic and state transitions
   - SFL (Simplified Fantasy Land) heuristic policy
   - Pybind11 bindings for Python integration
   - CUDA-accelerated operations

2. **Python Infrastructure**
   - `ofc_env.py` - Game environment wrapper
   - `ofc_types.py` - Type definitions
   - `ofc_scoring.py` - Scoring utilities
   - `state_encoding.py` - State encoding for neural networks

3. **SFL Heuristics**
   - `simulate_sfl_stats.py` - Run SFL simulations and get statistics
   - `sample_sfl_hands.py` - Generate sample hands played by SFL
   - `sfl_trace_one_hand.py` - Trace a single hand step-by-step
   - `tune_sfl.py` - Tune SFL reward shaping parameters
   - `sfl_param_sweep.py` - Parameter sweep for SFL tuning
   - `sfl_interactive.py` - Interactive SFL hand inspector

4. **RL Training Engine**
   - `rl_policy_net.py` - Policy network architecture
   - `train_rl_policy.py` - REINFORCE training script
   - `evaluate_rl_policy.py` - Evaluate trained policies

## Quick Start

### Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install numpy tqdm pybind11
   ```

3. **Build C++ extension:**
   ```bash
   cd cpp
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   cd ../..
   ```

   The compiled module will be `ofc_cpp.cp312-win_amd64.pyd` (Windows) or `ofc_cpp*.so` (Linux).

### Usage

#### Run SFL Heuristics

Evaluate SFL performance:
```bash
python simulate_sfl_stats.py --episodes 2000 --seed 999
```

Sample hands:
```bash
python sample_sfl_hands.py --hands 10
```

Tune SFL parameters:
```bash
python tune_sfl.py
```

#### Train RL Policy

Train a new policy:
```bash
python train_rl_policy.py --episodes 5000 --output models/rl_policy.pth
```

Evaluate a trained policy:
```bash
python evaluate_rl_policy.py --checkpoint models/rl_policy.pth --episodes 2000
```

#### Interactive Tools

Step through a hand interactively:
```bash
python sfl_interactive.py
```

Trace a single hand:
```bash
python sfl_trace_one_hand.py --hands 1
```

## Project Structure

```
.
├── cpp/                    # C++ game engine
│   ├── ofc_actions.cpp    # Action generation
│   ├── ofc_cpp.cpp        # Pybind11 bindings
│   ├── ofc_encode.cpp     # State encoding
│   ├── ofc_engine.cpp     # Game engine
│   ├── ofc_score.cpp      # Scoring logic
│   ├── ofc_step.cpp       # State transitions
│   ├── sfl_policy.cpp     # SFL heuristic
│   └── sfl_policy.h       # SFL header
├── ofc_env.py             # Python game environment
├── ofc_types.py           # Type definitions
├── ofc_scoring.py         # Scoring utilities
├── state_encoding.py      # Neural network encoding
├── rl_policy_net.py       # Policy network
├── train_rl_policy.py     # RL training
├── evaluate_rl_policy.py  # Policy evaluation
├── simulate_sfl_stats.py  # SFL statistics
├── sample_sfl_hands.py     # Sample SFL hands
├── sfl_trace_one_hand.py   # Hand tracing
├── tune_sfl.py            # SFL parameter tuning
├── sfl_interactive.py      # Interactive inspector
└── models/                # Trained models
```

## SFL Heuristics

The SFL (Simplified Fantasy Land) heuristic uses Monte Carlo rollouts to estimate action values. It can be tuned via reward shaping parameters:

- `foul_penalty`: Penalty for fouling (default: -4.0)
- `pass_penalty`: Penalty for passing (default: -3.0)
- `medium_bonus`: Bonus for medium royalties (default: 4.0)
- `strong_bonus`: Bonus for strong royalties (default: 8.0)
- `monster_mult`: Multiplier for monster hands (default: 10.0)

Use `tune_sfl.py` to experiment with different parameter values.

## RL Training

The RL system uses REINFORCE with:
- Moving average baseline for variance reduction
- Entropy regularization for exploration
- Canonical game scoring for rewards

Training is done on-policy, generating episodes and updating the policy network based on observed rewards.

## License

[Your License Here]
