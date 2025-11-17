#!/bin/bash
# Fix pybind11 installation on Linux

set -e

echo "============================================================"
echo "Checking and Fixing pybind11 Installation"
echo "============================================================"
echo

# Activate venv if not already
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
        echo "✓ Activated virtual environment"
    else
        echo "❌ Virtual environment not found!"
        exit 1
    fi
else
    echo "✓ Virtual environment already active"
fi

echo
echo "Current pybind11 status:"
python -c "import pybind11; print(f'  Version: {pybind11.__version__}')" 2>/dev/null || echo "  ❌ pybind11 not found"

echo
echo "Checking required packages..."
pip show pybind11 > /dev/null 2>&1 && echo "  ✓ pybind11 installed" || echo "  ❌ pybind11 NOT installed"
pip show scikit-build-core > /dev/null 2>&1 && echo "  ✓ scikit-build-core installed" || echo "  ❌ scikit-build-core NOT installed"
pip show setuptools > /dev/null 2>&1 && echo "  ✓ setuptools installed" || echo "  ❌ setuptools NOT installed"

echo
echo "Reinstalling pybind11 and build dependencies..."
pip uninstall -y pybind11 scikit-build-core 2>/dev/null || true
pip install --no-cache-dir pybind11>=2.11.0 scikit-build-core>=0.7.0 setuptools>=68

echo
echo "Verifying installation:"
python -c "import pybind11; print(f'  ✓ pybind11 {pybind11.__version__} installed')"
python -c "import scikit_build_core; print(f'  ✓ scikit-build-core installed')"

echo
echo "============================================================"
echo "pybind11 fix complete!"
echo "============================================================"
echo
echo "Next steps:"
echo "  1. Rebuild C++ extension:"
echo "     rm -rf build ofc_cpp*.so"
echo "     cmake -B build -S . -DCMAKE_BUILD_TYPE=Release"
echo "     cmake --build build --config Release"
echo "     cmake --install build --config Release || cp build/cpp/Release/ofc_cpp*.so ."
echo
echo "  2. Test: python test_architecture.py"

