# Rebuild Instructions for Linux Server

The C++ extension must be rebuilt after code changes. Run these commands:

```bash
cd ~/pineappleRLH200x8
git pull origin main
source .venv/bin/activate

# Clean old build artifacts
rm -rf build
rm -f ofc_cpp*.so
rm -f CMakeCache.txt CMakeFiles/

# Rebuild C++ extension
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build --config Release --clean-first
cmake --install build --config Release || cp build/cpp/Release/ofc_cpp*.so .

# Verify the extension loads
python -c "import ofc_cpp; print('✓ C++ extension loaded')"

# Run tests
python test_architecture.py
```

**Important:** After pulling code changes, you MUST rebuild the C++ extension. Python code changes don't require a rebuild, but C++ changes (in `cpp/` directory) do.
