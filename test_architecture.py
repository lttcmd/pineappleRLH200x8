"""
Comprehensive test to verify the full C++ workers + GPU training architecture.

Tests:
1. C++ environment logic (rules, state transitions, legal moves, scoring)
2. C++ workers running in parallel
3. pybind11 exposure to Python
4. Python batching states
5. GPU (PyTorch) evaluation
6. Python sending actions back to C++
7. Complete data flow: C++ → Python batches → GPU → Python → C++
"""

import torch
import numpy as np
import time
from typing import List, Tuple
import sys

# Test imports
try:
    import ofc_cpp as _CPP
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("WARNING: C++ extension not available. Some tests will be skipped.")

from value_net import ValueNet
from state_encoding import get_input_dim


def test_cpp_extension():
    """Test 1: Verify C++ extension is loaded via pybind11"""
    print("\n" + "="*60)
    print("TEST 1: C++ Extension (pybind11)")
    print("="*60)
    
    if not CPP_AVAILABLE:
        print("❌ FAILED: C++ extension not available")
        return False
    
    print("✓ C++ extension loaded successfully")
    
    # Test basic functions exist
    required_functions = [
        'generate_random_episodes',
        'create_engine',
        'destroy_engine',
        'engine_start_envs',
        'request_policy_batch',
        'apply_policy_actions',
        'engine_collect_encoded_episodes',
        'legal_actions_round0',
        'legal_actions_rounds1to4',
        'step_state',
        'step_state_round0',
        'score_board_from_ints',
        'encode_state_batch_ints'
    ]
    
    missing = []
    for func_name in required_functions:
        if not hasattr(_CPP, func_name):
            missing.append(func_name)
    
    if missing:
        print(f"❌ FAILED: Missing functions: {missing}")
        return False
    
    print(f"✓ All {len(required_functions)} required C++ functions available")
    return True


def test_cpp_game_logic():
    """Test 2: Verify C++ game rules, state transitions, legal moves, scoring"""
    print("\n" + "="*60)
    print("TEST 2: C++ Game Logic (Rules, Transitions, Legal Moves, Scoring)")
    print("="*60)
    
    if not CPP_AVAILABLE:
        print("⚠ SKIPPED: C++ extension not available")
        return True
    
    try:
        # Test 2a: Legal moves generation
        print("\n2a. Testing legal moves generation...")
        board_empty = np.full(13, -1, dtype=np.int16)
        placements = _CPP.legal_actions_round0(board_empty)
        assert placements.shape[0] > 0, "Should have legal placements for round 0"
        print(f"  ✓ Round 0: {placements.shape[0]} legal placements")
        
        # Test 2b: State transitions
        print("\n2b. Testing state transitions...")
        board = np.full(13, -1, dtype=np.int16)
        deck = np.arange(52, dtype=np.int16)
        np.random.shuffle(deck)
        current5 = deck[:5].copy()
        deck_remaining = deck[5:].copy()
        slots5 = np.array([0, 1, 2, 3, 4], dtype=np.int16)  # Place in first 5 slots
        
        result = _CPP.step_state_round0(board, current5, deck_remaining, slots5)
        new_board, new_round, new_draw, new_deck, done = result
        assert new_round == 1, "Should advance to round 1"
        assert new_draw.shape[0] == 3, "Should have 3 cards in next draw"
        print("  ✓ State transition successful")
        
        # Test 2c: Scoring
        print("\n2c. Testing scoring...")
        # Create a simple valid board
        bottom = np.array([0, 4, 8, 12, 16], dtype=np.int16)  # Some cards
        middle = np.array([20, 24, 28, 32, 36], dtype=np.int16)
        top = np.array([40, 44, 48], dtype=np.int16)
        score, fouled = _CPP.score_board_from_ints(bottom, middle, top)
        assert isinstance(score, (int, float)), "Score should be numeric"
        print(f"  ✓ Scoring works (score: {score:.2f}, fouled: {fouled})")
        
        print("\n✓ All C++ game logic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpp_parallel_workers():
    """Test 3: Verify C++ workers can run in parallel"""
    print("\n" + "="*60)
    print("TEST 3: C++ Parallel Workers")
    print("="*60)
    
    if not CPP_AVAILABLE:
        print("⚠ SKIPPED: C++ extension not available")
        return True
    
    try:
        print("\n3a. Testing parallel random episode generation...")
        num_episodes = 100
        seed = 12345
        
        start_time = time.time()
        encoded, offsets, scores = _CPP.generate_random_episodes(np.uint64(seed), num_episodes)
        elapsed = time.time() - start_time
        
        assert len(scores) == num_episodes, f"Should generate {num_episodes} episodes"
        assert encoded.shape[0] > 0, "Should have encoded states"
        # Offsets may be a list (new code) or array (old code)
        if isinstance(offsets, (list, tuple)):
            assert len(offsets) == num_episodes + 1, f"Should have correct offsets length, got {len(offsets)}"
            offsets_np = np.array(offsets, dtype=np.int32)
        else:
            assert offsets.shape[0] == num_episodes + 1, "Should have correct offsets"
            offsets_np = np.array(offsets, dtype=np.int32)
        
        episodes_per_sec = num_episodes / elapsed
        print(f"  ✓ Generated {num_episodes} episodes in {elapsed:.2f}s ({episodes_per_sec:.1f} eps/s)")
        
        print("\n3b. Testing engine with multiple environments...")
        h = _CPP.create_engine(np.uint64(seed))
        try:
            num_envs = 32
            _CPP.engine_start_envs(h, num_envs)
            
            # Request policy batch (should get candidates from multiple envs)
            enc, meta = _CPP.request_policy_batch(h, max_candidates_per_env=10)
            assert enc.shape[0] > 0, "Should get candidates from parallel envs"
            assert meta.shape[0] == enc.shape[0], "Should have meta for each candidate"
            
            print(f"  ✓ Engine created {num_envs} parallel environments")
            print(f"  ✓ Generated {enc.shape[0]} candidate states from parallel envs")
            
        finally:
            _CPP.destroy_engine(h)
        
        print("\n✓ All parallel worker tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_python_batching():
    """Test 4: Verify Python can batch states from C++"""
    print("\n" + "="*60)
    print("TEST 4: Python Batching States from C++")
    print("="*60)
    
    if not CPP_AVAILABLE:
        print("⚠ SKIPPED: C++ extension not available")
        return True
    
    try:
        # Generate states from C++
        num_episodes = 50
        encoded, offsets, scores = _CPP.generate_random_episodes(np.uint64(42), num_episodes)
        
        # Batch them in Python
        batch_size = 32
        num_states = encoded.shape[0]
        
        batches = []
        for i in range(0, num_states, batch_size):
            batch = encoded[i:i+batch_size]
            batches.append(torch.from_numpy(batch).float())
        
        total_batched = sum(b.shape[0] for b in batches)
        assert total_batched == num_states, "Should batch all states"
        
        print(f"  ✓ Batched {num_states} states into {len(batches)} batches")
        print(f"  ✓ Batch sizes: {[b.shape[0] for b in batches]}")
        
        print("\n✓ Python batching works correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_evaluation():
    """Test 5: Verify GPU (PyTorch) can evaluate batches"""
    print("\n" + "="*60)
    print("TEST 5: GPU (PyTorch) Evaluation")
    print("="*60)
    
    if not CPP_AVAILABLE:
        print("⚠ SKIPPED: C++ extension not available")
        return True
    
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n5a. Using device: {device}")
        
        # Create model
        input_dim = get_input_dim()
        model = ValueNet(input_dim, hidden_dim=512)
        model = model.to(device)
        model.eval()
        
        # Generate states from C++
        encoded, offsets, scores = _CPP.generate_random_episodes(np.uint64(99), 10)
        batch = torch.from_numpy(encoded[:32]).float().to(device)
        
        # Evaluate on GPU
        with torch.no_grad():
            values, foul_logit, round0_logits, feas_logit = model(batch)
        
        assert values.shape[0] == batch.shape[0], "Should output values for each state"
        assert values.shape[1] == 1, "Value should be scalar"
        
        print(f"  ✓ Model evaluated {batch.shape[0]} states on {device}")
        print(f"  ✓ Output shapes: values={values.shape}, foul_logit={foul_logit.shape}")
        
        # Test batch processing speed
        if device.type == "cuda":
            num_batches = 10
            batch_size = 128
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_batches):
                    test_batch = torch.randn(batch_size, input_dim, device=device)
                    _ = model(test_batch)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            batches_per_sec = num_batches / elapsed
            print(f"  ✓ GPU throughput: {batches_per_sec:.1f} batches/sec ({batch_size * batches_per_sec:.0f} states/sec)")
        
        print("\n✓ GPU evaluation works correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_data_flow():
    """Test 6: Complete data flow C++ → Python batches → GPU → Python → C++"""
    print("\n" + "="*60)
    print("TEST 6: Complete Data Flow")
    print("C++ generates states → Python batches → GPU evaluates → Python sends actions → C++ steps")
    print("="*60)
    
    if not CPP_AVAILABLE:
        print("⚠ SKIPPED: C++ extension not available")
        return True
    
    try:
        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = get_input_dim()
        model = ValueNet(input_dim, hidden_dim=512)
        model = model.to(device)
        model.eval()
        
        # Step 1: C++ generates states (via engine)
        print("\n6a. C++ generating candidate states...")
        seed = 12345
        h = _CPP.create_engine(np.uint64(seed))
        try:
            num_envs = 8
            _CPP.engine_start_envs(h, num_envs)
            
            # Request policy batch from C++
            enc, meta = _CPP.request_policy_batch(h, max_candidates_per_env=5)
            assert enc.shape[0] > 0, "Should get candidates from C++"
            print(f"  ✓ C++ generated {enc.shape[0]} candidate states")
            
            # Step 2: Python batches them
            print("\n6b. Python batching states...")
            batch = torch.from_numpy(enc).float().to(device)
            print(f"  ✓ Batched {batch.shape[0]} states, shape: {batch.shape}")
            
            # Step 3: GPU evaluates
            print("\n6c. GPU evaluating batch...")
            with torch.no_grad():
                values, foul_logit, _, _ = model(batch)
                values = values.squeeze()
                foul_prob = torch.sigmoid(foul_logit).squeeze()
                penalty = 8.0
                combined = values - penalty * foul_prob
                vals_cpu = combined.cpu().numpy()
            print(f"  ✓ GPU evaluated {batch.shape[0]} states")
            print(f"  ✓ Value range: [{vals_cpu.min():.2f}, {vals_cpu.max():.2f}]")
            
            # Step 4: Python selects best actions and sends back to C++
            print("\n6d. Python selecting actions and sending to C++...")
            best_by_env = {}
            for i in range(meta.shape[0]):
                env_id = int(meta[i, 0])
                action_id = int(meta[i, 1])
                v = vals_cpu[i]
                if (env_id not in best_by_env) or (v > best_by_env[env_id][0]):
                    best_by_env[env_id] = (v, action_id)
            
            if best_by_env:
                chosen = np.array([[e, a] for e, (_, a) in best_by_env.items()], dtype=np.int32)
                print(f"  ✓ Selected {len(chosen)} actions for {len(best_by_env)} environments")
                
                # Step 5: C++ steps environments
                print("\n6e. C++ stepping environments...")
                stepped = _CPP.apply_policy_actions(h, chosen)
                print(f"  ✓ C++ stepped {stepped} environments")
                
                # Collect results
                enc2, offs, scores = _CPP.engine_collect_encoded_episodes(h)
                if scores.shape[0] > 0:
                    print(f"  ✓ Collected {scores.shape[0]} completed episodes")
                    print(f"  ✓ Score range: [{scores.min():.2f}, {scores.max():.2f}]")
            
            print("\n✓ Complete data flow verified!")
            return True
            
        finally:
            _CPP.destroy_engine(h)
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop_integration():
    """Test 7: Verify training loop integration works"""
    print("\n" + "="*60)
    print("TEST 7: Training Loop Integration")
    print("="*60)
    
    if not CPP_AVAILABLE:
        print("⚠ SKIPPED: C++ extension not available")
        return True
    
    try:
        from train import SelfPlayTrainer
        from value_net import ValueNet
        from state_encoding import get_input_dim
        
        # Create a small trainer
        input_dim = get_input_dim()
        model = ValueNet(input_dim, hidden_dim=128)  # Smaller for testing
        trainer = SelfPlayTrainer(
            model=model,
            buffer_size=1000,
            batch_size=32,
            learning_rate=1e-3,
            use_cuda=torch.cuda.is_available(),
            num_workers=4
        )
        
        print("\n7a. Testing episode generation with C++ workers...")
        # Verify trainer can use C++
        from train import _USE_CPP, _CPP as TRAINER_CPP
        print(f"  Trainer C++ available: {_USE_CPP and TRAINER_CPP is not None}")
        
        # Generate a few episodes
        try:
            episodes = trainer.generate_episodes_parallel(num_episodes=10, use_random=True, base_seed=42)
        except Exception as e:
            print(f"  ❌ Error generating episodes: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Note: generate_episodes_parallel returns a flat list of (state, score) tuples
        # where states are encoded arrays when using C++
        if len(episodes) == 0:
            print(f"  ⚠ WARNING: generate_episodes_parallel returned empty list")
            print(f"  Trying direct C++ call to debug...")
            # Try direct C++ call to see what's happening
            try:
                encoded, offsets, scores = _CPP.generate_random_episodes(np.uint64(42), 10)
                # Offsets may be a list (new code) or array (old code)
                offsets_type = type(offsets).__name__
                if isinstance(offsets, (list, tuple)):
                    offsets_shape = f"list({len(offsets)})"
                    print(f"  Direct C++ call: encoded.shape={encoded.shape}, offsets={offsets_shape}, scores.shape={scores.shape}")
                    print(f"  Raw offsets (Python list): {offsets[:11] if len(offsets) >= 11 else offsets}")
                else:
                    offsets_shape = offsets.shape if hasattr(offsets, 'shape') else 'unknown'
                    print(f"  Direct C++ call: encoded.shape={encoded.shape}, offsets.shape={offsets_shape}, scores.shape={scores.shape}")
                    print(f"  Raw offsets (before conversion): {offsets}")
                    print(f"  Raw offsets type: {offsets_type}")
                    if hasattr(offsets, '__array__'):
                        print(f"  Raw offsets as array: {np.asarray(offsets)}")
                if scores.shape[0] > 0:
                    print(f"  C++ returned {scores.shape[0]} episodes, but trainer returned empty list")
                    print(f"  This suggests an issue in generate_episodes_parallel processing")
                    # Try to manually process like the trainer does
                    encoded_np = np.array(encoded, copy=False)
                    # Try with copy=True to see if that helps
                    offsets_np_copy = np.array(offsets, copy=True).astype(np.int32)
                    offsets_np = np.array(offsets, copy=False).astype(np.int32, copy=False)
                    print(f"  After conversion (copy=False): {offsets_np[:11] if len(offsets_np) >= 11 else offsets_np}")
                    print(f"  After conversion (copy=True): {offsets_np_copy[:11] if len(offsets_np_copy) >= 11 else offsets_np_copy}")
                    scores_np = np.array(scores, copy=False).astype(np.float32, copy=False)
                    print(f"  After conversion: encoded_np.shape={encoded_np.shape}, offsets_np.shape={offsets_np.shape}, scores_np.shape={scores_np.shape}")
                    print(f"  First few offsets: {offsets_np[:5] if len(offsets_np) >= 5 else offsets_np}")
                    print(f"  First few scores: {scores_np[:5] if len(scores_np) >= 5 else scores_np}")
                    # Try the processing loop
                    test_data = []
                    for e in range(scores_np.shape[0]):
                        s0 = int(offsets_np[e])
                        s1 = int(offsets_np[e+1])
                        score = float(scores_np[e])
                        print(f"  Episode {e}: s0={s0}, s1={s1}, score={score}, states={s1-s0}")
                        for s in range(s0, s1):
                            if s < encoded_np.shape[0]:
                                test_data.append((encoded_np[s].copy(), score))
                    print(f"  Manual processing created {len(test_data)} items")
            except Exception as e2:
                print(f"  Direct C++ call also failed: {e2}")
                import traceback
                traceback.print_exc()
        
        assert len(episodes) > 0, f"Should generate episodes, got {len(episodes)}"
        print(f"  ✓ Generated {len(episodes)} episode states using C++ workers")
        
        # Verify structure
        if len(episodes) > 0:
            first_item = episodes[0]
            assert len(first_item) == 2, "Each episode item should be (state, score) tuple"
            state_or_encoded, score = first_item
            assert isinstance(score, (int, float)), "Score should be numeric"
            print(f"  ✓ First episode: score={score:.2f}, state type={type(state_or_encoded).__name__}")
        
        print("\n7b. Testing training step...")
        # Add to buffer
        # Note: episodes is a flat list of (state, score) tuples
        num_to_add = min(50, len(episodes))
        for state_or_encoded, score in episodes[:num_to_add]:  # Add some to buffer
            # add_to_buffer expects a list of (state, score) tuples
            trainer.replay_buffer.append((state_or_encoded, score))
        
        # Try a training step
        if len(trainer.replay_buffer) >= trainer.batch_size:
            loss = trainer.train_step()
            assert isinstance(loss, (int, float)), "Should return loss value"
            print(f"  ✓ Training step completed, loss: {loss:.4f}")
        else:
            print(f"  ⚠ Buffer too small ({len(trainer.replay_buffer)} < {trainer.batch_size}), skipping training step")
        
        trainer.cleanup()
        print("\n✓ Training loop integration works")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all architecture tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ARCHITECTURE TEST")
    print("="*70)
    print("\nTesting full C++ workers + GPU training architecture:")
    print("  • C++ environment logic (rules, transitions, legal moves, scoring)")
    print("  • C++ workers running in parallel")
    print("  • pybind11 exposure to Python")
    print("  • Python batching states")
    print("  • GPU (PyTorch) evaluation")
    print("  • Complete data flow: C++ → Python → GPU → Python → C++")
    print("="*70)
    
    tests = [
        ("C++ Extension (pybind11)", test_cpp_extension),
        ("C++ Game Logic", test_cpp_game_logic),
        ("C++ Parallel Workers", test_cpp_parallel_workers),
        ("Python Batching", test_python_batching),
        ("GPU Evaluation", test_gpu_evaluation),
        ("Complete Data Flow", test_complete_data_flow),
        ("Training Loop Integration", test_training_loop_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{status:12} {test_name}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Architecture is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

