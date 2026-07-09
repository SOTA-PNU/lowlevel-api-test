# Enhanced Triton Dockerfile with CUDA Version Detection and GPU Architecture Support
# Automatically detects host CUDA version and builds accordingly
# Supports CUDA, CPU-only, and out-of-tree NPU backend builds

# +++ NPU: base changed 24.04 -> 22.04 because rebel-compiler ships ONLY cp310
#     (Python 3.10) wheels. 22.04 (py3.10, glibc 2.35) still supports upstream
#     Triton / torch / CUDA 12.8 toolkit, so cpu/gpu keep working. This is the one
#     unavoidable shared change; all cpu/gpu logic below is otherwise unchanged.
FROM ubuntu:22.04

# Build arguments for CUDA version detection (must be after FROM)
ARG BUILD_MODE=cuda
ARG CUDA_VERSION=12.8
ARG CUDA_ARCHITECTURE=sm_120
ARG PYTORCH_CUDA_VERSION=cu128
ARG NPU_PIP_PACKAGES=
ARG RBLN_COMPILER_VERSION=0.10.0   # +++ NPU: rebel-compiler (KMD 3.0.0 -> 0.10.x)

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
    libdrm2 \
    && rm -rf /var/lib/apt/lists/*
# +++ NPU: libdrm2 is REQUIRED by the npu runtime (librbln-thunk talks to the NPU
#     via DRM). Harmless for cpu/gpu. Without it, NPU access SEGFAULTs at runtime.

# Debug: Print BUILD_MODE value
RUN echo "DEBUG: BUILD_MODE value is: '$BUILD_MODE'"

# Install NVIDIA CUDA toolkit (only if BUILD_MODE is cuda)
RUN if [ "$BUILD_MODE" = "cuda" ]; then \
        echo "Installing CUDA toolkit..." && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update && \
        CUDA_PACKAGE_VERSION=$(echo "$CUDA_VERSION" | tr . -) && \
        apt-get install -y "cuda-toolkit-${CUDA_PACKAGE_VERSION}" && \
        rm cuda-keyring_1.1-1_all.deb && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping CUDA installation for ${BUILD_MODE} build"; \
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
RUN git submodule update --init --recursive

# Copy Triton source to a separate location to avoid mount conflicts
# GPU: upstream Triton from ./triton
# CPU: triton-cpu fork
# NPU: rebel.triton is vendored later; upstream source is not built
RUN if [ "$BUILD_MODE" = "cpu" ]; then \
        echo "Cloning triton-cpu source for CPU backend..." && \
        git clone https://github.com/triton-lang/triton-cpu.git /opt/triton-src && \
        cd /opt/triton-src && \
        git submodule update --init --recursive; \
    else \
        cp -r /workspace/triton /opt/triton-src; \
    fi

# Create virtual environment in /opt to avoid conflicts with host venv
RUN python3 -m venv /opt/triton-venv

# Upgrade pip and install basic requirements
RUN /opt/triton-venv/bin/pip install --upgrade pip setuptools wheel

# Install PyTorch with appropriate CUDA support based on build mode
RUN if [ "$BUILD_MODE" = "cuda" ]; then \
        /opt/triton-venv/bin/pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION}" && \
        /opt/triton-venv/bin/pip install filelock typing-extensions networkx jinja2 fsspec sympy==1.13.1 mpmath pillow; \
    elif [ "$BUILD_MODE" = "npu" ]; then \
        /opt/triton-venv/bin/pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu; \
    else \
        /opt/triton-venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-deps && \
        /opt/triton-venv/bin/pip install filelock typing-extensions networkx jinja2 fsspec sympy==1.13.1 mpmath pillow; \
    fi

# Install additional requirements
RUN /opt/triton-venv/bin/pip install numpy pytest

# +++ NPU: rebel.triton comes from rebel-compiler. Instead of DOWNLOADING it from
#     pypi.rbln.ai (which needs auth / a BuildKit secret), we VENDOR the host's
#     already-installed rebel-compiler from the build context -> NO credentials,
#     NO secret. Its PUBLIC dependencies are installed normally (public PyPI).
#     The exact public deps come from rebel-compiler's own METADATA (Requires-Dist).
RUN if [ "$BUILD_MODE" = "npu" ]; then \
        /opt/triton-venv/bin/pip install \
            attrs cloudpickle cryptography decorator ml_dtypes numpy packaging "prettytable>=3.8" \
            psutil scipy tornado tqdm typing_extensions; \
    fi

# +++ NPU: bake librbln user-space libs AND the vendored rebel-compiler packages.
#     docker/rbln-runtime/ MUST exist (COPY always runs; use .gitkeep for cpu/gpu).
#     Stage it on the RBLN host/runner (once, or via the CI step) with BOTH:
#       mkdir -p docker/rbln-runtime
#       cp -a /usr/lib/librbln-*.so* docker/rbln-runtime/
#       cp -a /usr/local/lib/python3.10/dist-packages/rebel \
#             /usr/local/lib/python3.10/dist-packages/tvm \
#             /usr/local/lib/python3.10/dist-packages/rebel_compiler.libs \
#             /usr/local/lib/python3.10/dist-packages/rebel_compiler-*.dist-info \
#             docker/rbln-runtime/
COPY docker/rbln-runtime/ /opt/rbln-lib/
RUN if [ "$BUILD_MODE" = "npu" ]; then \
        cp -a /opt/rbln-lib/librbln-*.so* /usr/lib/ && ldconfig && \
        cp -a /opt/rbln-lib/rebel /opt/rbln-lib/tvm /opt/rbln-lib/rebel_compiler.libs \
              /opt/rbln-lib/rebel_compiler-*.dist-info \
              /opt/triton-venv/lib/python3.10/site-packages/ && \
        /opt/triton-venv/bin/python -c "import rebel; print('vendored rebel-compiler', rebel.__version__)"; \
    fi

# Install Triton build dependencies
RUN if [ "$BUILD_MODE" != "npu" ]; then \
        /opt/triton-venv/bin/pip install -r /opt/triton-src/python/requirements.txt; \
    fi

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
    echo 'elif [ "$BUILD_MODE" = "npu" ]; then' >> /workspace/detect_and_build.sh && \
    echo '    echo "📋 NPU build mode detected"' >> /workspace/detect_and_build.sh && \
    echo '    CUDA_ARCH="none"' >> /workspace/detect_and_build.sh && \
    echo '    echo "🏗️  Building Triton with out-of-tree NPU backend support"' >> /workspace/detect_and_build.sh && \
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
    echo 'elif [ "$BUILD_MODE" = "npu" ]; then' >> /workspace/detect_and_build.sh && \
    echo '    echo "🚀 Building Triton for NPU backend execution"' >> /workspace/detect_and_build.sh && \
    echo 'else' >> /workspace/detect_and_build.sh && \
    echo '    echo "🚀 Building Triton for CPU-only execution"' >> /workspace/detect_and_build.sh && \
    echo 'fi' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
        echo '# Build/install selected Triton source' >> /workspace/detect_and_build.sh && \
    echo 'echo "🏗️  Building/installing selected Triton source..."' >> /workspace/detect_and_build.sh && \
    echo 'cd /opt/triton-src' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo 'if [ "$BUILD_MODE" = "cpu" ]; then' >> /workspace/detect_and_build.sh && \
    echo '    echo "🏗️  Installing triton-cpu from source..."' >> /workspace/detect_and_build.sh && \
    echo '    export TRITON_CPU_BACKEND=1' >> /workspace/detect_and_build.sh && \
    echo '    /opt/triton-venv/bin/pip install -e . --no-build-isolation' >> /workspace/detect_and_build.sh && \
    echo 'elif [ "$BUILD_MODE" = "cuda" ]; then' >> /workspace/detect_and_build.sh && \
    echo '    echo "🏗️  Installing upstream Triton from source..."' >> /workspace/detect_and_build.sh && \
    echo '    /opt/triton-venv/bin/pip install -e .' >> /workspace/detect_and_build.sh && \
    echo '    /opt/triton-venv/bin/python setup.py build_ext --inplace' >> /workspace/detect_and_build.sh && \
    echo 'else' >> /workspace/detect_and_build.sh && \
    echo '    echo "🧠 NPU build uses rebel.triton; skipping upstream Triton build."' >> /workspace/detect_and_build.sh && \
    echo 'fi' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo 'echo "✅ Triton source install/build step completed"' >> /workspace/detect_and_build.sh && \
    echo '' >> /workspace/detect_and_build.sh && \
    echo '# Verify installation' >> /workspace/detect_and_build.sh && \
    echo 'echo "🧪 Verifying Triton installation..."' >> /workspace/detect_and_build.sh && \
    echo 'if [ "$BUILD_MODE" = "cpu" ]; then TRITON_CPU_BACKEND=1 /opt/triton-venv/bin/python -c "import triton; print(\"✅ triton-cpu imported successfully\")"; elif [ "$BUILD_MODE" = "cuda" ]; then /opt/triton-venv/bin/python -c "import triton; import torch; print(\"✅ Triton imported successfully\"); print(f\"CUDA available: {torch.cuda.is_available()}\")"; else echo "NPU uses rebel.triton verification later"; fi' >> /workspace/detect_and_build.sh

RUN chmod +x /workspace/detect_and_build.sh

# Run the detection and build script with BUILD_MODE environment variable
# +++ NPU: skip the upstream-Triton build for npu (it uses rebel.triton instead).
RUN if [ "$BUILD_MODE" != "npu" ]; then BUILD_MODE=$BUILD_MODE /workspace/detect_and_build.sh; fi

# Keep --local-triton imports pointed at the built source tree.
# +++ NPU: for npu just remove the unused upstream triton source (avoids a
#     namespace-shadow on `import triton`); no symlink needed.
RUN if [ "$BUILD_MODE" != "npu" ]; then \
        rm -rf /workspace/triton && ln -s /opt/triton-src /workspace/triton; \
    else \
        rm -rf /workspace/triton; \
    fi

# Verify the build artifacts/backend registration for the selected build mode.
RUN if [ "$BUILD_MODE" = "cuda" ]; then \
        ls -la /opt/triton-src/python/triton/_C/ || echo "Build artifacts verification failed"; \
    elif [ "$BUILD_MODE" = "npu" ]; then \
        /opt/triton-venv/bin/python -c "import rebel; print('rebel-compiler', rebel.__version__); from rebel.triton.backends import backends; ks=sorted(backends); print('rebel.triton backends:', ks); assert any(k.lower()=='rebel' for k in ks), 'rebel Triton backend not registered in the image'; print('OK: rebel Triton backend present')"; \
    elif [ "$BUILD_MODE" = "cpu" ]; then \
        TRITON_CPU_BACKEND=1 /opt/triton-venv/bin/python -c "import triton; print('OK: triton-cpu import works')"; \
    fi

# Set default environment
ENV VIRTUAL_ENV="/opt/triton-venv"
ENV PATH="/opt/triton-venv/bin:$PATH"

# Expose port for Jupyter if needed
EXPOSE 8888

# Default command - run tests (will auto-detect device)
CMD ["/opt/triton-venv/bin/python", "triton_test.py", "--device", "auto"]
