## Windows Setup Guide (venv, dependencies, C++ build)

This guide assumes:
- Windows 10/11
- Python 3.9+ installed and on your `PATH`
- CMake and a C++ compiler (Visual Studio Build Tools) for building the C++ extension

All commands are shown for **PowerShell**.

---

### 1. Install system prerequisites

1. **Install Python 3.9+**  
   - Download from `https://www.python.org/downloads/windows/`  
   - During install, check “Add Python to PATH”.

2. **Install Visual Studio Build Tools (C++ compiler)**  
   - Install “Build Tools for Visual Studio” from `https://visualstudio.microsoft.com/downloads/`.  
   - During setup, select the “Desktop development with C++” workload.

3. **Install CMake** (if you don’t have it)  
   - Download from `https://cmake.org/download/` and install.  
   - Ensure `cmake` is on your `PATH` (`cmake --version` should work in PowerShell).

---

### 2. Clone the repository

```powershell
cd C:\path\to\where\you\want\the\repo
git clone https://github.com/lttcmd/pineappleRLH200x8.git
cd pineappleRLH200x8
```

If you already have the repo:

```powershell
cd C:\path\to\pineappleRLH200x8
git pull origin main
```

---

### 3. Create and activate a virtual environment

From the repo root (where `train.py` is):

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate
```

Verify:

```powershell
python --version
pip --version
```

You should see the Python from `.venv`.

---

### 4. Install PyTorch


#### 4.2 CUDA 12.1 (modern NVIDIA GPU)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4.3 CUDA 11.8 (older NVIDIA GPU)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'version:', torch.version.cuda)"
```

---

### 5. Install Python dependencies

With the venv activated:

```powershell
pip install -r requirements.txt

pip install pybind11 scikit-build-core setuptools

pip install pytest
```

These cover:
- `numpy`, `tqdm`, `matplotlib`, `tensorboard`
- build tools for the C++ extension
- `pytest` for running tests

---

### 6. Build the C++ extension (`ofc_cpp`)

From the repo root:

```powershell
REM Clean previous build (optional)
if (Test-Path build) { Remove-Item -Recurse -Force build }

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD

cmake --build build --config Release

cmake --install build --config Release
```

After a successful build/install, you should have an extension file in the project root, e.g.:
- `ofc_cpp.cp3XX-win_amd64.pyd`

Verify that Python can import it:

```powershell
python -c "import ofc_cpp; print('✓ ofc_cpp imported from', ofc_cpp.__file__)"
```

If install fails but the compiled `.pyd` exists under `build\cpp\Release`, you can copy it manually:

```powershell
Copy-Item build\cpp\Release\ofc_cpp*.pyd .
```

---

### 7. Run architecture and basic tests

With venv active and in the repo root:

#### 7.1 Comprehensive architecture test

```powershell
python test_architecture.py
```

This verifies:
- C++ extension loads and exposes functions.
- C++ game logic, parallel workers, batching.
- GPU evaluation and full data flow.

#### 7.2 Pytest suite

```powershell
pytest
```

This runs:
- `tests/test_env_round0.py`
- `tests/test_scoring_equivalence.py`
- `tests/test_parallel_generation.py`
- `tests/test_checkpoint_schedule.py`
and more.

---

### 8. Start training or experiments

#### 8.1 Legacy RL training

```powershell
python train.py
```

This uses the current `SelfPlayTrainer` with `ValueNet`.

#### 8.2 Supervised pretraining (optional)

```powershell
python supervised_train.py
```

This builds a dataset of “good/neutral/foul” boards and trains a pure value model, saving:
- `supervised_value_net.pth`
- `value_net_checkpoint_ep0.pth`

#### 8.3 Two-player head-to-head evaluation

```powershell
python two_player_sim.py --checkpoint value_net_checkpoint_ep0.pth --matches 50
```

This plays your model vs a random baseline from the same deck.

---

### 9. Common issues

- **`ModuleNotFoundError: No module named 'ofc_cpp'`**  
  - Ensure the `.pyd` lives in the repo root and that the build completed.  
  - Re-run the C++ build steps and the import test.

- **CMake not found**  
  - Ensure `cmake` is on PATH or install from `cmake.org`.

- **Visual C++ not found**  
  - Install “Desktop development with C++” via Visual Studio Build Tools and restart PowerShell.


