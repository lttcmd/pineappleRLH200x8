# Fix PyTorch Installation on Linux

If you see errors like:
```
OSError: cannot open shared object file: No such file or directory
```

Your PyTorch installation is corrupted or incomplete.

## Quick Fix

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Uninstall PyTorch completely
pip uninstall torch torchvision torchaudio -y

# 3. Clear pip cache
pip cache purge

# 4. Reinstall PyTorch (choose one based on your setup)

# Option A: CPU-only (if no GPU)
pip install torch torchvision torchaudio

# Option B: CUDA 12.1 (if you have NVIDIA GPU with CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Option C: CUDA 11.8 (if you have NVIDIA GPU with CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Check Your GPU Setup First

Before reinstalling, check if you have a GPU:

```bash
# Check for NVIDIA GPU
nvidia-smi

# If that works, you have a GPU - use CUDA version
# If it fails, use CPU-only version
```

## After Reinstalling

Once PyTorch is fixed, run the test again:

```bash
python test_architecture.py
```

