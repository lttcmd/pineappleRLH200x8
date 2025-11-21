#!/bin/bash
# Quick setup script for Linux server
# Run with: bash setup_linux_server.sh

set -e  # Exit on error

echo "=== Linux Server Setup Script ==="
echo ""

# Check if running as root - adjust sudo usage accordingly
if [ "$EUID" -eq 0 ]; then 
   echo "Running as root - will skip sudo for package management"
   SUDO=""
else
   SUDO="sudo"
fi

echo "Step 1: Updating system packages..."
$SUDO apt update
$SUDO apt upgrade -y

echo ""
echo "Step 2: Installing build tools..."
$SUDO apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl

echo ""
echo "Step 3: Checking Python version..."
python3 --version

echo ""
echo "Step 4: Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo ""
echo "Step 5: Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Step 6: Installing Python dependencies..."
pip install torch torchvision torchaudio
pip install numpy tqdm pybind11

echo ""
echo "Step 7: Building C++ extension..."
cd cpp
mkdir -p build
cd build
cmake ..
cmake --build . --config Release -j$(nproc)
cd ../..

echo ""
echo "Step 8: Copying compiled module to root..."
find . -name "ofc_cpp*.so" -exec cp {} . \; 2>/dev/null || echo "Note: .so file location may vary"

echo ""
echo "Step 9: Testing C++ module import..."
python3 -c "import ofc_cpp; print('✓ C++ module loaded successfully!')" || {
    echo "⚠ Warning: Could not import ofc_cpp. You may need to manually copy the .so file."
}

echo ""
echo "Step 10: Testing SFL heuristics..."
python3 simulate_sfl_stats.py --episodes 5 --seed 42 || {
    echo "⚠ Warning: SFL test failed. Check the error above."
}

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "CPU cores available: $(nproc)"
echo ""
echo "To generate 10M dataset, run:"
echo "  source .venv/bin/activate"
echo "  python3 generate_sfl_dataset.py \\"
echo "    --output data/sfl_dataset_10m \\"
echo "    --examples-per-shard 1000 \\"
echo "    --num-shards 10000 \\"
echo "    --seed 42 \\"
echo "    --num-workers $(nproc) \\"
echo "    --fast"
echo ""

