#!/usr/bin/env python3
"""
Verify all dependencies are correctly installed and working.
"""

import sys
import importlib

def check_module(name, import_name=None):
    """Check if a module can be imported."""
    if import_name is None:
        import_name = name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {name}: NOT FOUND - {e}")
        return False
    except Exception as e:
        print(f"⚠ {name}: ERROR - {e}")
        return False

def check_pybind11():
    """Check pybind11 specifically."""
    try:
        import pybind11
        print(f"✓ pybind11: {pybind11.__version__}")
        
        # Check if numpy integration works (pybind11 includes numpy support)
        import numpy as np
        # pybind11 doesn't have a separate numpy module, it's integrated
        print("  ✓ pybind11 numpy integration available")
        return True
    except Exception as e:
        print(f"❌ pybind11: ERROR - {e}")
        return False

def check_cpp_extension():
    """Check C++ extension."""
    try:
        import ofc_cpp
        print(f"✓ ofc_cpp: Loaded successfully")
        
        # Check if key functions exist
        required_funcs = [
            'generate_random_episodes',
            'legal_actions_round0',
            'step_state',
            'score_board_from_ints'
        ]
        missing = []
        for func in required_funcs:
            if not hasattr(ofc_cpp, func):
                missing.append(func)
        
        if missing:
            print(f"  ⚠ Missing functions: {missing}")
            return False
        else:
            print(f"  ✓ All required functions available")
            return True
    except ImportError as e:
        print(f"❌ ofc_cpp: NOT FOUND - {e}")
        print("  → Run: cmake --build build --config Release")
        return False
    except Exception as e:
        print(f"❌ ofc_cpp: ERROR - {e}")
        return False

def check_pytorch():
    """Check PyTorch and CUDA."""
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  ✓ CUDA available: {torch.version.cuda}")
            print(f"  ✓ GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠ CUDA not available (CPU-only mode)")
        
        # Check if can create tensors
        x = torch.tensor([1.0, 2.0, 3.0])
        if x.shape[0] == 3:
            print(f"  ✓ Tensor creation works")
        
        return True
    except Exception as e:
        print(f"❌ torch: ERROR - {e}")
        return False

def check_numpy():
    """Check NumPy."""
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
        
        # Check array operations
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        if arr.shape[0] == 5:
            print(f"  ✓ Array operations work")
        
        return True
    except Exception as e:
        print(f"❌ numpy: ERROR - {e}")
        return False

def check_cmake():
    """Check CMake version."""
    import subprocess
    try:
        result = subprocess.run(['cmake', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✓ cmake: {version}")
            return True
        else:
            print(f"❌ cmake: Not found")
            return False
    except FileNotFoundError:
        print(f"❌ cmake: Not installed")
        return False
    except Exception as e:
        print(f"⚠ cmake: ERROR - {e}")
        return False

def main():
    print("=" * 60)
    print("DEPENDENCY VERIFICATION")
    print("=" * 60)
    print()
    
    results = []
    
    print("Python Environment:")
    print(f"  Python: {sys.version}")
    print()
    
    print("Build Tools:")
    results.append(("cmake", check_cmake()))
    print()
    
    print("Python Packages:")
    results.append(("numpy", check_numpy()))
    results.append(("pybind11", check_pybind11()))
    results.append(("torch", check_pytorch()))
    results.append(("tqdm", check_module("tqdm")))
    results.append(("matplotlib", check_module("matplotlib")))
    print()
    
    print("C++ Extension:")
    results.append(("ofc_cpp", check_cpp_extension()))
    print()
    
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ All dependencies verified!")
        return 0
    else:
        print("⚠ Some dependencies have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())

