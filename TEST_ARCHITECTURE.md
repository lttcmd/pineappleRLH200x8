# Architecture Test Guide

## Overview

The `test_architecture.py` file comprehensively tests the entire C++ workers + GPU training architecture to verify:

1. ✅ **C++ Extension (pybind11)** - All C++ functions exposed to Python
2. ✅ **C++ Game Logic** - Rules, state transitions, legal moves, scoring
3. ✅ **C++ Parallel Workers** - Multiple environments running in parallel
4. ✅ **Python Batching** - States batched from C++ for GPU processing
5. ✅ **GPU Evaluation** - PyTorch model evaluating batches on GPU
6. ✅ **Complete Data Flow** - Full cycle: C++ → Python → GPU → Python → C++
7. ✅ **Training Loop Integration** - End-to-end training with C++ workers

## Running the Test

### Basic Usage

```bash
# Activate virtual environment first
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Run the test
python test_architecture.py
```

### Expected Output

The test will run all 7 test suites and print:
- ✓ or ❌ for each test
- Detailed output for each component
- Performance metrics (episodes/sec, batches/sec)
- Final summary with pass/fail counts

### Example Output

```
======================================================================
COMPREHENSIVE ARCHITECTURE TEST
======================================================================

Testing full C++ workers + GPU training architecture:
  • C++ environment logic (rules, transitions, legal moves, scoring)
  • C++ workers running in parallel
  • pybind11 exposure to Python
  • Python batching states
  • GPU (PyTorch) evaluation
  • Complete data flow: C++ → Python → GPU → Python → C++
======================================================================

============================================================
TEST 1: C++ Extension (pybind11)
============================================================
✓ C++ extension loaded successfully
✓ All 13 required C++ functions available

============================================================
TEST 2: C++ Game Logic (Rules, Transitions, Legal Moves, Scoring)
============================================================
...
[Detailed test output]
...

======================================================================
TEST SUMMARY
======================================================================
✓ PASSED     C++ Extension (pybind11)
✓ PASSED     C++ Game Logic
✓ PASSED     C++ Parallel Workers
✓ PASSED     Python Batching
✓ PASSED     GPU Evaluation
✓ PASSED     Complete Data Flow
✓ PASSED     Training Loop Integration
======================================================================
Total: 7/7 tests passed

🎉 All tests passed! Architecture is working correctly.
```

## What Each Test Verifies

### Test 1: C++ Extension
- Verifies `ofc_cpp` module loads
- Checks all required functions are available
- Confirms pybind11 binding works

### Test 2: C++ Game Logic
- Legal moves generation (round 0 and rounds 1-4)
- State transitions (stepping through game)
- Scoring function (validates board scoring)

### Test 3: C++ Parallel Workers
- `generate_random_episodes()` generates multiple episodes
- Engine creates multiple parallel environments
- `request_policy_batch()` gets candidates from parallel envs

### Test 4: Python Batching
- States from C++ can be batched in Python
- Batch sizes are correct
- All states are included

### Test 5: GPU Evaluation
- Model can evaluate batches on GPU (or CPU)
- Output shapes are correct
- GPU throughput is measured

### Test 6: Complete Data Flow
- **C++ generates** candidate states via engine
- **Python batches** them for GPU
- **GPU evaluates** the batch
- **Python selects** best actions
- **C++ steps** environments with selected actions
- **C++ collects** results

### Test 7: Training Loop Integration
- `SelfPlayTrainer` can generate episodes with C++ workers
- Training step works with batched data
- End-to-end training loop functions

## Troubleshooting

### "C++ extension not available"
- Build the C++ extension first (see SETUP_INSTRUCTIONS.md)
- Verify `ofc_cpp.pyd` (Windows) or `ofc_cpp.so` (Linux/Mac) exists

### "CUDA not available"
- Test will still run on CPU
- GPU tests will be skipped or run on CPU
- Install PyTorch with CUDA if you want GPU testing

### Individual test failures
- Check the detailed error output
- Verify C++ extension is built correctly
- Ensure all Python dependencies are installed

## Success Criteria

All 7 tests must pass to confirm the architecture is working:

1. ✅ C++ extension loads and exposes all functions
2. ✅ C++ game logic works correctly
3. ✅ C++ can run parallel workers
4. ✅ Python can batch C++ states
5. ✅ GPU can evaluate batches
6. ✅ Complete data flow works end-to-end
7. ✅ Training loop integrates everything

If all tests pass, your architecture is correctly set up! 🎉

