#!/bin/bash

# Enhanced Triton Docker Build Script with CUDA Version Detection
# Automatically detects host CUDA version and GPU architecture
# Supports both CUDA and CPU-only builds

set -e

echo "🐳 Building Enhanced Triton Docker Image with CUDA Detection..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "⚠️  NVIDIA Docker runtime not detected. CUDA support may not work."
fi

# Detect CUDA availability
echo "🔍 Detecting CUDA availability..."
CUDA_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    CUDA_AVAILABLE=true
    echo "✅ CUDA detected on host system"
else
    echo "⚠️  CUDA not detected on host system - will build CPU-only version"
fi

# Set build mode based on CUDA availability
if [ "$CUDA_AVAILABLE" = true ]; then
    BUILD_MODE="cuda"
    # Detect host CUDA version
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K\d+\.\d+' | head -1)
    echo "📋 Host CUDA version: $CUDA_VERSION"
    # Use CUDA 12.6 for Ubuntu 24.04 compatibility
    CUDA_VERSION="12.6"
    echo "📋 Using CUDA toolkit version: $CUDA_VERSION (Ubuntu 24.04 compatible)"
else
    BUILD_MODE="cpu"
    CUDA_VERSION="none"
    echo "📋 Building CPU-only version (no CUDA)"
fi

# Detect GPU architecture (only if CUDA is available)
if [ "$BUILD_MODE" = "cuda" ]; then
    echo "🎮 Detecting GPU architecture..."
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "📋 Host GPU: $GPU_NAME"
    
    # Map GPU names to CUDA architectures
    case "$GPU_NAME" in
        *"Blackwell"*|*"RTX PRO 6000"*)
            CUDA_ARCH="sm_120"
            PYTORCH_CUDA_VERSION="cu124"  # Use latest PyTorch for Blackwell
            echo "🏗️  Detected Blackwell architecture: $CUDA_ARCH"
            ;;
        *"Hopper"*|*"H100"*)
            CUDA_ARCH="sm_90"
            PYTORCH_CUDA_VERSION="cu121"
            echo "🏗️  Detected Hopper architecture: $CUDA_ARCH"
            ;;
        *"Ampere"*|*"A100"*|*"RTX 30"*|*"RTX 40"*)
            CUDA_ARCH="sm_80"
            PYTORCH_CUDA_VERSION="cu121"
            echo "🏗️  Detected Ampere architecture: $CUDA_ARCH"
            ;;
        *"Turing"*|*"RTX 20"*|*"GTX 16"*)
            CUDA_ARCH="sm_75"
            PYTORCH_CUDA_VERSION="cu121"
            echo "🏗️  Detected Turing architecture: $CUDA_ARCH"
            ;;
        *)
            CUDA_ARCH="sm_80"
            PYTORCH_CUDA_VERSION="cu121"
            echo "🏗️  Unknown GPU, using default architecture: $CUDA_ARCH"
            ;;
    esac
else
    CUDA_ARCH="none"
    PYTORCH_CUDA_VERSION="cpu"
    echo "🏗️  CPU-only build - no GPU architecture needed"
fi

# Remove existing image if it exists
# if docker image inspect triton-local-build:latest > /dev/null 2>&1; then
#     echo "🗑️  Removing existing triton-local-build:latest image..."
#     docker rmi triton-local-build:latest
# fi

# Build the Docker image with detected parameters
if [ "$BUILD_MODE" = "cuda" ]; then
    echo "📦 Building Docker image with CUDA $CUDA_VERSION and architecture $CUDA_ARCH..."
else
    echo "📦 Building Docker image in CPU-only mode (no CUDA)..."
fi
echo "⏳ This may take 30-60 minutes depending on your system..."

docker build \
    --cache-from triton-local-build:latest \
    --build-arg BUILD_MODE=$BUILD_MODE \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg CUDA_ARCHITECTURE=$CUDA_ARCH \
    --build-arg PYTORCH_CUDA_VERSION=$PYTORCH_CUDA_VERSION \
    -t triton-local-build:latest \
    -f docker/Dockerfile .

docker image prune -f

echo "✅ Enhanced Docker build completed successfully!"
echo ""
echo "📊 Build Summary:"
echo "  Build Mode: $BUILD_MODE"
echo "  CUDA Version: $CUDA_VERSION"
echo "  GPU Architecture: $CUDA_ARCH"
echo "  PyTorch CUDA: $PYTORCH_CUDA_VERSION"
echo ""
echo "🚀 Available commands:"
echo "  ./run-docker.sh test        - Run Triton tests"
echo "  ./run-docker.sh dev         - Start development container"
echo "  ./run-docker.sh jupyter     - Start Jupyter Lab"
echo "  ./run-docker.sh bash        - Start interactive bash session"
echo ""
if [ "$BUILD_MODE" = "cuda" ]; then
    echo "💡 The enhanced build automatically detects your GPU architecture"
    echo "   and builds Triton with the appropriate CUDA support."
else
    echo "💡 CPU-only build completed. Triton will run on CPU without CUDA support."
    echo "   To enable CUDA support, install NVIDIA drivers and CUDA toolkit."
fi

echo ""
echo "=========================================="
echo "BUILD_COMPLETED_SUCCESSFULLY"
echo "✅ TRITON BUILD COMPLETED SUCCESSFULLY ✅"
echo "=========================================="
echo "Build Mode: $BUILD_MODE"
echo "CUDA Version: $CUDA_VERSION"
echo "GPU Architecture: $CUDA_ARCH"
echo "PyTorch CUDA: $PYTORCH_CUDA_VERSION"
echo ""
echo "🚀 Next step: ./docker/run-docker.sh test"
echo "=========================================="