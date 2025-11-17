# Linux Server Setup Guide (Digital Ocean)

Complete setup instructions for running the project on a Linux server.

## Prerequisites

- Ubuntu/Debian Linux server
- Python 3.9+ installed
- CMake 3.18+ installed
- C++ compiler (g++ or clang)
- Git installed

## Step-by-Step Setup Commands

### 1. Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install Python, pip, and build tools
sudo apt-get install -y python3 python3-pip python3-venv

# Install CMake and C++ build tools
sudo apt-get install -y cmake build-essential

# Install Git (if not already installed)
sudo apt-get install -y git
```

### 2. Clone/Pull the Repository

```bash
# If cloning fresh:
git clone https://github.com/lttcmd/pineappleRLH200x8.git
cd pineappleRLH200x8

# OR if you already have it, just pull:
cd pineappleRLH200x8
git pull origin main
```

### 3. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify you're in the venv (should show (.venv) in prompt)
which python
```

### 4. Install PyTorch (CPU or CUDA)

**For CPU-only (if no GPU):**
```bash
pip install torch torchvision torchaudio
```

**For CUDA (if you have GPU with CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8 (alternative):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU availability:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### 5. Install Python Dependencies

```bash
# Make sure venv is activated
source .venv/bin/activate

# Install required packages
pip install numpy
pip install tqdm
pip install matplotlib
pip install tensorboard

# Install build dependencies for C++ extension
pip install pybind11 scikit-build-core setuptools
```

### 6. Build C++ Extension with CMake

```bash
# Make sure venv is activated
source .venv/bin/activate

# Clean any old build (optional)
rm -rf build

# Configure CMake build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)

# Build the extension
cmake --build build --config Release

# Install/copy the extension to project root
cmake --install build --config Release

# OR manually copy if install fails:
# cp build/cpp/Release/ofc_cpp*.so .  # Linux uses .so, not .pyd
```

**Note:** On Linux, the extension will be `ofc_cpp.cpython-*.so` instead of `.pyd`

### 7. Verify Installation

```bash
# Make sure venv is activated
source .venv/bin/activate

# Test C++ extension loads
python -c "import ofc_cpp; print('✓ C++ extension loaded successfully')"

# Run architectural test
python test_architecture.py
```

### 8. (Optional) Run Learning Test

```bash
# Make sure venv is activated
source .venv/bin/activate

# Test that model can learn
python test_learning.py
```

## Quick Setup Script (All-in-One)

You can also create a setup script:

```bash
#!/bin/bash
# save as setup.sh, then: chmod +x setup.sh && ./setup.sh

set -e  # Exit on error

echo "=== Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv cmake build-essential git

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing PyTorch ==="
# Uncomment the one you need:
pip install torch torchvision torchaudio
# OR for CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing Python dependencies ==="
pip install numpy tqdm matplotlib tensorboard pybind11 scikit-build-core setuptools

echo "=== Building C++ extension ==="
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build --config Release
cmake --install build --config Release || cp build/cpp/Release/ofc_cpp*.so .

echo "=== Verifying installation ==="
python -c "import ofc_cpp; print('✓ C++ extension loaded')"
python test_architecture.py

echo "=== Setup complete! ==="
echo "To activate environment: source .venv/bin/activate"
```

## Running Training

Once setup is complete:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start training
python train.py
```

## Troubleshooting

**If CMake fails:**
```bash
# Check CMake version (need 3.18+)
cmake --version

# If too old, install newer version or use pip:
pip install cmake
```

**If C++ extension doesn't load:**
```bash
# Check if .so file exists
ls -la ofc_cpp*.so

# Check Python can find it
python -c "import sys; print(sys.path)"
```

**If GPU not detected:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version
```

## Notes

- On Linux, the C++ extension is `.so` (shared object) instead of `.pyd` (Windows)
- Make sure to activate the venv before running any Python commands
- If you have a GPU, install the appropriate CUDA version of PyTorch
- The build directory can be deleted after installation (the `.so` file is copied to project root)

