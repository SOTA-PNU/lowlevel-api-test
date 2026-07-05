# Enhanced Triton Dockerfile with CUDA Version Detection and GPU Architecture Support
# Automatically detects host CUDA version and builds accordingly
# Supports both CUDA and CPU-only builds

# Use Ubuntu 24.04 base image and install CUDA manually
FROM ubuntu:24.04

# Build arguments for CUDA version detection (must be after FROM)
ARG BUILD_MODE=cuda
ARG CUDA_VERSION=12.6
ARG CUDA_ARCHITECTURE=sm_120
ARG PYTORCH_CUDA_VERSION=cu121

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV PATH=/opt/triton-venv/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies including CUDA
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    vim \
    pkg-config \
    libffi-dev \
    libssl-dev \
    gnupg2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Debug: Print BUILD_MODE value
RUN echo "DEBUG: BUILD_MODE value is: '$BUILD_MODE'"

# Install NVIDIA CUDA toolkit (only if BUILD_MODE is cuda)
RUN if [ "$BUILD_MODE" = "cuda" ]; then \
        echo "Installing CUDA toolkit..." && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update && \
        apt-get install -y cuda-toolkit-12-6 && \
        rm cuda-keyring_1.1-1_all.deb && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping CUDA installation for CPU-only build"; \
    fi

# Set CUDA environment variables (only if BUILD_MODE is cuda)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /workspace

# Copy the project files
COPY . /workspace/

# Initialize and update git submodules
RUN git submodule update --init --recursive && \
    mv /workspace/triton /opt/triton-src && \
    ln -s /opt/triton-src /workspace/trito

# Copy Triton source to a separate location to avoid mount conflicts
# RUN cp -r /workspace/triton /opt/triton-src

# Create virtual environment in /opt to avoid conflicts with host venv
RUN python3 -m venv /opt/triton-venv

# Upgrade pip and install basic requirements
RUN /opt/triton-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with appropriate CUDA support based on build mode
RUN if [ "$BUILD_MODE" = "cuda" ]; then \
        /opt/triton-venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-deps && \
        /opt/triton-venv/bin/pip install --no-cache-dir filelock typing-extensions networkx jinja2 fsspec sympy==1.13.1 mpmath && \
        /opt/triton-venv/bin/pip install --no-cache-dir nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-cupti-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.21.5 nvidia-nvtx-cu12==12.1.105 pillow; \
    else \
        /opt/triton-venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-deps && \
        /opt/triton-venv/bin/pip install --no-cache-dir filelock typing-extensions networkx jinja2 fsspec sympy==1.13.1 mpmath pillow; \
    fi

# Install additional requirements
RUN /opt/triton-venv/bin/pip install --no-cache-dir numpy pytest

# Install Triton build dependencies
RUN /opt/triton-venv/bin/pip install --no-cache-dir -r /opt/triton-src/python/requirements.txt

# Create a script to detect GPU architecture and build Triton accordingly
RUN echo '#!/bin/bash' > /workspace/detect_and_build.sh && \
    echo 'set -e' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo 'echo "🔍 Detecting build mode and architecture..."' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '# Check build mode from environment' >> /workspace/detect_and_build.sh && \
    echo 'if [ "$BUILD_MODE" = "cpu" ]; then' >> /workspace/detect_and_build.sh && \
    echo '    echo "📋 CPU-only build mode detected"' >> /workspace/detect_and_build.sh && \
    echo '    CUDA_ARCH="none"' >> /workspace/detect_and_build.sh && \
    echo '    echo "🏗️  Building Triton for CPU-only execution"' >> /workspace/detect_and_build.sh && \
    echo 'else' >> /workspace/detect_and_build.sh && \
    echo '    echo "📋 CUDA build mode detected"' >> /workspace/detect_and_build.sh && \
    echo '    ' >> /workspace/detect_and_build.sh && \
    echo '    # Detect CUDA version' >> /workspace/detect_and_build.sh && \
    echo '    if command -v nvcc &> /dev/null; then' >> /workspace/detect_and_build.sh && \
    echo '        CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)' >> /workspace/detect_and_build.sh && \
    echo '        echo "📋 Detected CUDA version: $CUDA_VERSION"' >> /workspace/detect_and_build.sh && \
    echo '    else' >> /workspace/detect_and_build.sh && \
    echo '        echo "⚠️  nvcc not found, using default CUDA architecture"' >> /workspace/detect_and_build.sh && \
    echo '    fi' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '    # Detect GPU architecture' >> /workspace/detect_and_build.sh && \
    echo '    if command -v nvidia-smi &> /dev/null; then' >> /workspace/detect_and_build.sh && \
    echo '        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)' >> /workspace/detect_and_build.sh && \
    echo '        echo "🎮 Detected GPU: $GPU_NAME"' >> /workspace/detect_and_build.sh && \
    echo '        ' >> /workspace/detect_and_build.sh && \
    echo '        # Map GPU names to CUDA architectures' >> /workspace/detect_and_build.sh && \
    echo '        case "$GPU_NAME" in' >> /workspace/detect_and_build.sh && \
    echo '            *"Blackwell"*|*"RTX PRO 6000"*)' >> /workspace/detect_and_build.sh && \
    echo '                CUDA_ARCH="sm_120"' >> /workspace/detect_and_build.sh && \
    echo '                echo "🏗️  Using CUDA architecture: $CUDA_ARCH (Blackwell)"' >> /workspace/detect_and_build.sh && \
    echo '                ;;' >> /workspace/detect_and_build.sh && \
    echo '            *"Hopper"*|*"H100"*)' >> /workspace/detect_and_build.sh && \
    echo '                CUDA_ARCH="sm_90"' >> /workspace/detect_and_build.sh && \
    echo '                echo "🏗️  Using CUDA architecture: $CUDA_ARCH (Hopper)"' >> /workspace/detect_and_build.sh && \
    echo '                ;;' >> /workspace/detect_and_build.sh && \
    echo '            *"Ampere"*|*"A100"*|*"RTX 30"*|*"RTX 40"*)' >> /workspace/detect_and_build.sh && \
    echo '                CUDA_ARCH="sm_80"' >> /workspace/detect_and_build.sh && \
    echo '                echo "🏗️  Using CUDA architecture: $CUDA_ARCH (Ampere)"' >> /workspace/detect_and_build.sh && \
    echo '                ;;' >> /workspace/detect_and_build.sh && \
    echo '            *"Turing"*|*"RTX 20"*|*"GTX 16"*)' >> /workspace/detect_and_build.sh && \
    echo '                CUDA_ARCH="sm_75"' >> /workspace/detect_and_build.sh && \
    echo '                echo "🏗️  Using CUDA architecture: $CUDA_ARCH (Turing)"' >> /workspace/detect_and_build.sh && \
    echo '                ;;' >> /workspace/detect_and_build.sh && \
    echo '            *"Volta"*|*"V100"*)' >> /workspace/detect_and_build.sh && \
    echo '                CUDA_ARCH="sm_70"' >> /workspace/detect_and_build.sh && \
    echo '                echo "🏗️  Using CUDA architecture: $CUDA_ARCH (Volta)"' >> /workspace/detect_and_build.sh && \
    echo '                ;;' >> /workspace/detect_and_build.sh && \
    echo '            *)' >> /workspace/detect_and_build.sh && \
    echo '                CUDA_ARCH="sm_80"  # Default to Ampere' >> /workspace/detect_and_build.sh && \
    echo '                echo "🏗️  Unknown GPU, using default CUDA architecture: $CUDA_ARCH"' >> /workspace/detect_and_build.sh && \
    echo '                ;;' >> /workspace/detect_and_build.sh && \
    echo '        esac' >> /workspace/detect_and_build.sh && \
    echo '    else' >> /workspace/detect_and_build.sh && \
    echo '        CUDA_ARCH="sm_80"  # Default fallback' >> /workspace/detect_and_build.sh && \
    echo '        echo "🏗️  nvidia-smi not available, using default CUDA architecture: $CUDA_ARCH"' >> /workspace/detect_and_build.sh && \
    echo '    fi' >> /workspace/detect_and_build.sh && \
    echo 'fi' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '# Set environment variables for Triton build' >> /workspace/detect_and_build.sh && \
    echo 'if [ "$BUILD_MODE" = "cuda" ]; then' >> /workspace/detect_and_build.sh && \
    echo '    export TRITON_CUDA_ARCHITECTURES="$CUDA_ARCH"' >> /workspace/detect_and_build.sh && \
    echo '    export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"' >> /workspace/detect_and_build.sh && \
    echo '    echo "🚀 Building Triton with CUDA architecture: $CUDA_ARCH"' >> /workspace/detect_and_build.sh && \
    echo 'else' >> /workspace/detect_and_build.sh && \
    echo '    echo "🚀 Building Triton for CPU-only execution"' >> /workspace/detect_and_build.sh && \
    echo 'fi' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '# Force build Triton from source (skip pip installation)' >> /workspace/detect_and_build.sh && \
    echo 'echo "🏗️  Building Triton from source (skipping pip installation)..."' >> /workspace/detect_and_build.sh && \
    echo 'cd /opt/triton-src' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '# Install Triton in development mode' >> /workspace/detect_and_build.sh && \
    echo '/opt/triton-venv/bin/pip install -e .' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '# Build extensions with proper architecture' >> /workspace/detect_and_build.sh && \
    echo '/opt/triton-venv/bin/python setup.py build_ext --inplace' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo 'echo "🧹 Cleaning up build artifacts..."' >> /workspace/detect_and_build.sh && \
    echo 'rm -rf build/ .eggs/ triton.egg-info/' >> /workspace/detect_and_build.sh && \
    echo 'echo "✅ Triton built successfully from source"' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '# Verify installation' >> /workspace/detect_and_build.sh && \
    echo 'echo "🧪 Verifying Triton installation..."' >> /workspace/detect_and_build.sh && \
    echo '/opt/triton-venv/bin/python -c "import triton; print(f\"✅ Triton imported successfully\"); import torch; print(f\"📋 PyTorch version: {torch.__version__}\"); print(f\"📋 CUDA available: {torch.cuda.is_available()}\"); print(\"📋 Running in CPU-only mode\" if not torch.cuda.is_available() else f\"📋 CUDA version: {torch.version.cuda}\")"' >> /workspace/detect_and_build.sh

RUN chmod +x /workspace/detect_and_build.sh

# Run the detection and build script with BUILD_MODE environment variable
RUN BUILD_MODE=$BUILD_MODE /workspace/detect_and_build.sh

# Keep --local-triton imports pointed at the built source tree.
# RUN rm -rf /workspace/triton && ln -s /opt/triton-src /workspace/triton

# Verify the build artifacts are present (only for CUDA builds)
RUN if [ "$BUILD_MODE" = "cuda" ]; then \
        ls -la /opt/triton-src/python/triton/_C/ || echo "Build artifacts verification failed"; \
    else \
        echo "CPU-only build - skipping CUDA artifacts verification"; \
    fi

# Set default environment
ENV VIRTUAL_ENV="/opt/triton-venv"
ENV PATH="/opt/triton-venv/bin:$PATH"

# Expose port for Jupyter if needed
EXPOSE 8888

# Default command - run tests (will auto-detect device)
CMD ["/opt/triton-venv/bin/python", "triton_test.py", "--device", "auto"]
