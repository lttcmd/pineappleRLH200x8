# Clean Rebuild Instructions for Linux

If you're seeing the offset bug on Linux (all offsets are the same value), you need to do a **complete clean rebuild**.

## The Problem

The offsets bug was fixed in commit `ceb841f`, but if your Linux build is using cached/old object files, it won't have the fix.

## Solution: Complete Clean Rebuild

Run these commands **in order** on your Linux server:

```bash
# 1. Make sure you're in the project directory
cd pineappleRLH200x8

# 2. Pull latest code (make sure you have the fix)
git pull origin main

# 3. Verify the fix is in the code
grep -A 5 "Offsets are already correctly built" cpp/ofc_engine.cpp
# Should show: "// Offsets are already correctly built as cumulative values during the episode loop"

# 4. Activate virtual environment
source .venv/bin/activate

# 5. COMPLETELY remove old build artifacts
rm -rf build
rm -f ofc_cpp*.so
rm -f ofc_cpp*.pyd
find . -name "*.so" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 6. Clean CMake cache (if it exists)
rm -rf CMakeCache.txt CMakeFiles/

# 7. Rebuild from scratch
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build --config Release --clean-first

# 8. Install/copy the extension
cmake --install build --config Release || cp build/cpp/Release/ofc_cpp*.so .

# 9. Verify the extension exists
ls -la ofc_cpp*.so

# 10. Test that it loads
python -c "import ofc_cpp; print('✓ C++ extension loaded')"

# 11. Run the architectural test
python test_architecture.py
```

## If It Still Fails

If you still see the bug after a clean rebuild, check:

1. **Verify the fix is in the code:**
   ```bash
   grep -n "offsets.push_back(state_count)" cpp/ofc_engine.cpp
   # Should show line 535 (inside the episode loop)
   
   grep -n "Always recalculate offsets" cpp/ofc_engine.cpp
   # Should return NOTHING (that buggy code was removed)
   ```

2. **Check git commit:**
   ```bash
   git log --oneline -1
   # Should show: ceb841f Fix episode offsets bug...
   ```

3. **Force rebuild with verbose output:**
   ```bash
   rm -rf build
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd) --verbose
   cmake --build build --config Release --verbose
   ```

## Expected Behavior After Fix

After the fix, offsets should be **cumulative**:
- Episode 0: offset 0 → 5
- Episode 1: offset 5 → 10  
- Episode 2: offset 10 → 15
- etc.

NOT all the same value like `[50, 50, 50, ...]`

