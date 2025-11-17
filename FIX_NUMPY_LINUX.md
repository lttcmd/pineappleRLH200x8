# Fix NumPy Installation Issue on Linux

If you see errors like:
```
ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'
ImportError: Error importing numpy
```

Your NumPy installation is corrupted. Here's how to fix it:

## Complete Fix (Recommended)

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Uninstall NumPy and related packages
pip uninstall numpy -y

# 3. Clear pip cache
pip cache purge

# 4. Reinstall NumPy
pip install numpy

# 5. Verify
python -c "import numpy; print('NumPy version:', numpy.__version__)"
```

## Nuclear Option: Reinstall All Dependencies

If the above doesn't work, reinstall everything:

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Uninstall all ML packages
pip uninstall torch torchvision torchaudio numpy -y

# 3. Clear pip cache
pip cache purge

# 4. Reinstall in order
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# OR for CPU-only: pip install torch torchvision torchaudio

# 5. Install other dependencies
pip install tqdm matplotlib tensorboard

# 6. Verify
python -c "import torch; import numpy; print('✓ All imports work')"
```

