"""
Test script to verify C++ path is being used.
"""
import os
# Enable C++ explicitly
os.environ['OFC_USE_CPP'] = '1'

from ofc_env import OfcEnv, State, Action
import sys

def test_cpp_path():
    """Test if C++ module is loaded and being used."""
    print("="*50)
    print("Testing C++ Path")
    print("="*50)
    
    # Check if C++ module is available
    try:
        import ofc_cpp
        print("[OK] C++ module (ofc_cpp) is loaded")
        print(f"  Module location: {ofc_cpp.__file__ if hasattr(ofc_cpp, '__file__') else 'builtin'}")
    except ImportError as e:
        print(f"[FAIL] C++ module NOT loaded: {e}")
        print("  Will use Python fallback path")
        return False
    
    # Check environment variable
    use_cpp = os.getenv("OFC_USE_CPP", "1") == "1"
    print(f"  OFC_USE_CPP environment variable: {use_cpp}")
    
    # Create environment
    env = OfcEnv()
    state = env.reset()
    
    print(f"\n  Initial state:")
    print(f"    Round: {state.round}")
    print(f"    Cards in draw: {len(state.current_draw)}")
    print(f"    Cards in deck: {len(state.deck)}")
    
    # Get legal actions
    legal_actions = env.legal_actions(state)
    print(f"    Legal actions: {len(legal_actions)}")
    
    if not legal_actions:
        print("  [FAIL] No legal actions!")
        return False
    
    # Take first action
    action = legal_actions[0]
    print(f"\n  Taking action: {action}")
    print(f"    Placements: {action.placements}")
    
    # Step and check if cards were placed
    cards_before = sum(1 for c in state.board if c is not None)
    print(f"    Cards on board before: {cards_before}/13")
    print(f"    Current draw: {[str(c) for c in state.current_draw]}")
    
    # Check what Python will pass to C++
    import numpy as np
    board_arr = np.array([c.to_int() if c is not None else -1 for c in state.board], dtype=np.int16)
    current5 = np.array([c.to_int() for c in state.current_draw], dtype=np.int16)
    slots5 = np.array([p[1] for p in action.placements], dtype=np.int16)
    print(f"    Python will pass to C++:")
    print(f"      board_arr: {board_arr}")
    print(f"      current5: {current5}")
    print(f"      slots5: {slots5}")
    
    state, reward, done = env.step(state, action)
    
    cards_after = sum(1 for c in state.board if c is not None)
    print(f"    Cards on board after: {cards_after}/13")
    print(f"    Board after: {[str(c) if c else 'None' for c in state.board]}")
    
    # Also check what C++ actually returned by calling it directly
    import ofc_cpp
    import numpy as np
    board_arr2 = np.array([c.to_int() if c is not None else -1 for c in state.board if True], dtype=np.int16)
    # Re-get state before step
    state2 = env.reset()
    board_arr_before = np.ascontiguousarray([c.to_int() if c is not None else -1 for c in state2.board], dtype=np.int16)
    current5_before = np.ascontiguousarray([c.to_int() for c in state2.current_draw], dtype=np.int16)
    slots5_before = np.ascontiguousarray([p[1] for p in action.placements], dtype=np.int16)
    print(f"    Array flags:")
    print(f"      board_arr contiguous: {board_arr_before.flags['C_CONTIGUOUS']}")
    print(f"      current5 contiguous: {current5_before.flags['C_CONTIGUOUS']}")
    print(f"      slots5 contiguous: {slots5_before.flags['C_CONTIGUOUS']}")
    try:
        result = ofc_cpp.step_state_round0(board_arr_before, current5_before, 
                                           np.array([c.to_int() for c in state2.deck], dtype=np.int16),
                                           slots5_before)
        b2_direct = result[0]
        print(f"\n    Direct C++ call result:")
        print(f"      Board returned: {b2_direct}")
        print(f"      Cards in returned board: {sum(1 for v in b2_direct if v >= 0)}")
        print(f"      First 5 slots: {b2_direct[:5]}")
        print(f"      Expected: cards {current5_before} in slots {slots5_before}")
    except Exception as e:
        print(f"\n    ERROR calling C++ directly: {e}")
        import traceback
        traceback.print_exc()
    
    if cards_after > cards_before:
        print("  [OK] Cards were placed - C++ path is working!")
        return True
    else:
        print("  [FAIL] Cards were NOT placed - C++ path may have issues")
        print("    This suggests the C++ step function isn't working correctly")
        return False

if __name__ == '__main__':
    success = test_cpp_path()
    sys.exit(0 if success else 1)

