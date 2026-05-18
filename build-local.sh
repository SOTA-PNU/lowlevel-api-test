#!/bin/bash

# Local Triton Build Script
# Builds Triton locally without Docker

set -e

echo "🔧 Building Triton locally..."

# Check if we're in the right directory
if [ ! -f "triton_test.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if triton submodule exists
if [ ! -d "triton" ]; then
    echo "❌ Triton submodule not found. Please run:"
    echo "   git submodule update --init --recursive"
    exit 1
fi

# Check Python requirements
echo "📋 Checking Python requirements..."
python3 -c "import sys; print(f'Python version: {sys.version}')"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Check if virtual environment was created successfully
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "   CUDA detected - installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio
else
    echo "   No CUDA detected - installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install basic requirements
echo "📦 Installing basic requirements..."
pip install numpy pytest

# Install triton pip version as fallback
echo "📦 Installing pip triton (fallback)..."
pip install triton || echo "   Triton pip install failed (this is OK)"

# Install Triton build dependencies
echo "📦 Installing Triton build dependencies..."
if [ -f "triton/python/requirements.txt" ]; then
    pip install -r triton/python/requirements.txt || echo "   Some dependencies failed (this might be OK)"
fi

# Build local Triton
echo "🔨 Building local Triton..."
cd triton

# Try building with pip install -e
echo "   Attempting pip install -e ."
if pip install -e .; then
    echo "✅ Local Triton build successful!"
    cd ..

    # Test the build
    echo "🧪 Testing local Triton build..."
    python triton_test.py --check-build

    echo ""
    echo "🚀 Build completed! You can now run:"
    echo "   source venv/bin/activate"
    echo "   python triton_test.py --local-triton"
else
    echo "❌ Local Triton build failed"
    cd ..

    echo ""
    echo "💡 You can still use pip-installed triton:"
    echo "   source venv/bin/activate"
    echo "   python triton_test.py"
fi

echo ""
echo "🎯 Available testing options:"
echo "   python triton_test.py --check-build      # Check build status"
echo "   python triton_test.py --device cpu       # CPU testing"
echo "   python triton_test.py --local-triton     # Use local build"
echo "   python triton_test.py                    # Use pip version"