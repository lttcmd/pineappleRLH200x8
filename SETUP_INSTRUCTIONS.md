# Step-by-Step Setup Instructions

## Prerequisites
- Python 3.8+ installed
- CMake 3.18+ installed
- C++ compiler (Visual Studio on Windows, or GCC/Clang on Linux/Mac)
- CUDA-capable GPU (optional but recommended for training)

## Step 1: Navigate to Project Directory
```bash
cd pineappleRL-Optimization
```

## Step 2: Create Python Virtual Environment
```bash
python -m venv .venv
```

## Step 3: Activate Virtual Environment

**On Windows:**
```bash
.venv\Scripts\activate
```

**On Linux/Mac:**
```bash
source .venv/bin/activate
```

## Step 4: Install Python Dependencies

### 4a. Install PyTorch with CUDA (if you have an NVIDIA GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**OR** if you don't have CUDA or want CPU-only:
```bash
pip install torch torchvision torchaudio
```

### 4b. Install Other Required Packages
```bash
pip install numpy tqdm matplotlib tensorboard pybind11
```

## Step 5: Verify GPU Availability (if using CUDA)
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

You should see: `CUDA available: True` (if GPU is available)

## Step 6: Build C++ Extension

### Option A: If `ofc_cpp.pyd` (Windows) or `ofc_cpp.so` (Linux/Mac) already exists
You can skip this step and go to Step 7.

### Option B: Build from scratch

**On Windows (using Visual Studio):**
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
```

**On Linux/Mac:**
```bash
mkdir build
cd build
cmake ..
make -j4
cd ..
```

The compiled extension should be in the root directory as:
- `ofc_cpp.pyd` (Windows)
- `ofc_cpp.so` (Linux/Mac)

## Step 7: Verify C++ Extension Works
```bash
python -c "import ofc_cpp; print('C++ extension loaded successfully!')"
```

If you see an error, the C++ extension needs to be rebuilt (go back to Step 6).

## Step 8: Run Training

### Basic Training (100,000 episodes)
```bash
python train.py
```

### What to Expect:
- Training will start and show progress bars
- Episodes are generated using C++ workers (much faster than before)
- Model checkpoints saved every 10,000 episodes as `value_net_checkpoint_ep*.pth`
- Final model saved as `value_net.pth`

### Training Output:
- Progress bar showing:
  - Episode count
  - Random exploration percentage
  - Royalty rate
  - Buffer size
  - Loss values (during training steps)

## Step 9: (Optional) Test the Trained Model

After training completes, you can test the model:
```bash
python demo.py
```

This will play one hand using the trained model.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ofc_cpp'"
**Solution:** The C++ extension isn't built. Go to Step 6 and build it.

### Issue: "CUDA not available" but you have an NVIDIA GPU
**Solution:** 
1. Make sure you installed PyTorch with CUDA support (Step 4a)
2. Verify your GPU drivers are up to date
3. Check: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: CMake errors during build
**Solution:**
1. Make sure CMake 3.18+ is installed: `cmake --version`
2. On Windows, make sure Visual Studio C++ build tools are installed
3. On Linux, install build essentials: `sudo apt-get install build-essential` (Ubuntu/Debian)

### Issue: Training is slow
**Solution:**
- Make sure the C++ extension is built and working (Step 7)
- Check that `OFC_USE_CPP=1` is set (it's the default)
- Verify you're using GPU: training should show "Using device: cuda"

## Architecture Overview

After this setup, your training will use:
- **C++ Workers**: All game logic (rules, state transitions, legal moves, scoring) runs in fast C++
- **Python Trainer**: Neural network training on GPU (PyTorch)
- **Data Flow**: C++ generates states → Python batches → GPU evaluates → Python sends actions → C++ steps

This architecture is much faster than the previous Python-only implementation!

