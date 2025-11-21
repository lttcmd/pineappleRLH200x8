# Linux Server Setup Guide

Complete guide to set up a fresh Linux server and generate 10M SFL dataset.

## Prerequisites

- Fresh Linux server (Ubuntu 20.04/22.04 recommended)
- Root/sudo access
- At least 96 CPU cores (for parallel generation)
- ~50GB free disk space (for 10M dataset)

## Step 1: Initial Server Setup

### 1.1 Update system packages
```bash
sudo apt update
sudo apt upgrade -y
```

### 1.2 Install essential build tools
```bash
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl
```

### 1.3 Verify Python version (need 3.8+)
```bash
python3 --version
# Should show Python 3.8 or higher
```

## Step 2: Clone Repository

```bash
# Navigate to your home directory or desired location
cd ~

# Clone the repository
git clone https://github.com/lttcmd/pineappleRLH200x8.git
cd pineappleRLH200x8
```

## Step 3: Set Up Python Environment

### 3.1 Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Upgrade pip
```bash
pip install --upgrade pip
```

### 3.3 Install PyTorch (CPU version for dataset generation)
```bash
# For CPU-only (dataset generation doesn't need GPU)
pip install torch torchvision torchaudio

# OR if you want CUDA support (for later training):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.4 Install other dependencies
```bash
pip install numpy tqdm pybind11
```

## Step 4: Build C++ Extension

### 4.1 Install CMake (if not already installed)
```bash
# Verify CMake version (need 3.18+)
cmake --version

# If version is too old, install newer version:
# wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.tar.gz
# tar -xzf cmake-3.28.0-linux-x86_64.tar.gz
# sudo mv cmake-3.28.0-linux-x86_64 /opt/cmake
# export PATH=/opt/cmake/bin:$PATH
```

### 4.2 Build the C++ module
```bash
# Build from root directory (CMakeLists.txt is in root)
mkdir -p build
cd build
cmake ..
cmake --build . --config Release -j$(nproc)
cd ..
```

### 4.3 Verify the build
```bash
# The compiled module should be in build/cpp/ or root directory
find build -name "ofc_cpp*.so" -exec cp {} . \; 2>/dev/null

# Check if it exists
ls -la *.so
# Should see something like: ofc_cpp.cpython-*.so

# Test import
python3 -c "import ofc_cpp; print('C++ module loaded successfully!')"
```

If you see an error about the module not found, you may need to:
```bash
# Copy the .so file to the root directory
find build -name "ofc_cpp*.so" -exec cp {} . \;
# Or search more broadly:
find . -name "ofc_cpp*.so" -type f
```

## Step 5: Verify Setup

### 5.1 Test SFL heuristics
```bash
python3 simulate_sfl_stats.py --episodes 10 --seed 42
```

You should see output like:
```json
{
  "episodes": 10,
  "fouls": 2,
  "passes": 4,
  "royalty_boards": 4,
  ...
}
```

### 5.2 Check CPU cores
```bash
nproc
# Should show 96 (or your actual core count)
```

## Step 6: Generate 10M Dataset

### 6.1 Create output directory
```bash
mkdir -p data/sfl_dataset_10m
```

### 6.2 Run dataset generation
```bash
python3 generate_sfl_dataset.py \
  --output data/sfl_dataset_10m \
  --examples-per-shard 1000 \
  --num-shards 10000 \
  --seed 42 \
  --num-workers 96 \
  --fast
```

**What this does:**
- Generates 10,000 shards × 1,000 examples = 10M total examples
- Uses all 96 CPU cores in parallel
- Uses fast SFL rollouts for speed
- Saves to `data/sfl_dataset_10m/`

**Expected time:** ~4-5 hours on 96 cores

**Progress:** You'll see a progress bar showing shards completed.

### 6.3 Monitor progress
```bash
# In another terminal, check how many shards are done:
ls data/sfl_dataset_10m/ | wc -l

# Check disk usage:
du -sh data/sfl_dataset_10m/
```

## Step 7: Verify Dataset

### 7.1 Check shard count
```bash
ls data/sfl_dataset_10m/ | wc -l
# Should show 10000
```

### 7.2 Quick test load
```bash
python3 -c "
import torch
d = torch.load('data/sfl_dataset_10m/shard_0000.pt', weights_only=False)
print('Keys:', list(d.keys()))
print('States:', d['encoded'].shape[0])
print('Actions:', d['action_encodings'].shape[0])
"
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ofc_cpp'"
**Solution:**
```bash
# Make sure you're in the repo root
pwd
# Should show: .../pineappleRLH200x8

# Check if .so file exists
ls -la *.so

# If not, rebuild:
cd cpp/build
cmake --build . --config Release
find . -name "ofc_cpp*.so" -exec cp {} ../../ \;
cd ../..
```

### Issue: "Permission denied" when building
**Solution:**
```bash
# Make sure you have write permissions
chmod -R u+w cpp/build
```

### Issue: Out of memory during generation
**Solution:**
- Reduce `--num-workers` (e.g., `--num-workers 48`)
- Each worker uses some memory, so 96 workers might be too many

### Issue: CMake version too old
**Solution:**
```bash
# Install newer CMake
wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.tar.gz
tar -xzf cmake-3.28.0-linux-x86_64.tar.gz
sudo mv cmake-3.28.0-linux-x86_64 /opt/cmake
export PATH=/opt/cmake/bin:$PATH
echo 'export PATH=/opt/cmake/bin:$PATH' >> ~/.bashrc
```

## Next Steps After Generation

Once the 10M dataset is generated:

1. **Train supervised model:**
   ```bash
   python3 train_supervised_sfl.py \
     --data data/sfl_dataset_10m \
     --output models/supervised_sfl_10m.pth \
     --epochs 10 \
     --batch-size 64
   ```

2. **Evaluate model:**
   ```bash
   python3 evaluate_rl_policy.py \
     --checkpoint models/supervised_sfl_10m.pth \
     --episodes 2000 \
     --seed 42
   ```

3. **Fine-tune with RL:**
   ```bash
   python3 train_rl_policy.py \
     --checkpoint models/supervised_sfl_10m.pth \
     --episodes 50000 \
     --output models/rl_finetuned_10m.pth
   ```

## Quick Reference Commands

```bash
# Activate environment
source .venv/bin/activate

# Check GPU (if installed)
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check CPU cores
nproc

# Monitor system resources
htop

# Check disk space
df -h

# Check dataset progress
ls data/sfl_dataset_10m/ | wc -l
```

