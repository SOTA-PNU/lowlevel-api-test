# Enhanced Triton Dockerfile with CUDA Version Detection and GPU Architecture Support
# Automatically detects host CUDA version and builds accordingly
# Supports CUDA, CPU-only, and out-of-tree NPU backend builds

# +++ NPU: base changed 24.04 -> 22.04 because rebel-compiler ships ONLY cp310
#     (Python 3.10) wheels. 22.04 (py3.10, glibc 2.35) still supports upstream
FROM ubuntu:22.04

# Build arguments for CUDA version detection (must be after FROM)
ARG BUILD_MODE=cuda
ARG CUDA_VERSION=12.8
ARG CUDA_ARCHITECTURE=sm_120
ARG PYTORCH_CUDA_VERSION=cu128
ARG NPU_PIP_PACKAGES=
# Last triton-cpu revision verified by a successful full CPU image build/test run.
ARG TRITON_CPU_REF=6ca45149d5b622f1d0c91f0ecf5000eb8224f489
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
        git checkout "$TRITON_CPU_REF" && \
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
#     already-installed rebel-compiler from the build context
RUN if [ "$BUILD_MODE" = "npu" ]; then \
        /opt/triton-venv/bin/pip install \
            attrs cloudpickle cryptography decorator ml_dtypes numpy packaging "prettytable>=3.8" \
            psutil scipy tornado tqdm typing_extensions; \
    fi

# +++ NPU: bake librbln user-space libs AND the vendored rebel-compiler packages.
COPY docker/rbln-runtime/ /opt/rbln-lib/
RUN if [ "$BUILD_MODE" = "npu" ]; then \
        cp -a /opt/rbln-lib/rbln-smi /opt/rbln-lib/rbln-stat /usr/local/bin/ && \
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

# Build/install the Triton implementation selected by BUILD_MODE directly.
# CUDA_VERSION and CUDA_ARCHITECTURE are already detected in stable_test.yml
# and passed to this Dockerfile through --build-arg, so no detect script is needed.
RUN if [ "$BUILD_MODE" = "cpu" ]; then \
        echo "Installing triton-cpu from source..." && \
        cd /opt/triton-src && \
        export TRITON_CPU_BACKEND=1 && \
        /opt/triton-venv/bin/pip install -e . --no-build-isolation && \
        TRITON_CPU_BACKEND=1 /opt/triton-venv/bin/python -c \
            "import triton; print('OK: triton-cpu imported successfully')"; \
    elif [ "$BUILD_MODE" = "cuda" ]; then \
        echo "Installing upstream Triton (runner architecture: ${CUDA_ARCHITECTURE})..." && \
        cd /opt/triton-src && \
        /opt/triton-venv/bin/pip install -e . --no-build-isolation && \
        /opt/triton-venv/bin/python -c \
            "import triton; print('OK: upstream Triton imported successfully')"; \
    elif [ "$BUILD_MODE" = "npu" ]; then \
        echo "NPU build uses vendored rebel.triton; skipping upstream Triton build."; \
    else \
        echo "ERROR: Unsupported BUILD_MODE=${BUILD_MODE}" >&2 && \
        exit 1; \
    fi

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
        /opt/triton-venv/bin/python -c "import triton; from triton.backends import backends; ks=sorted(backends); print('Triton backends:', ks); assert 'nvidia' in ks, 'NVIDIA Triton backend not registered in the image'; print('OK: NVIDIA Triton backend present')"; \
    elif [ "$BUILD_MODE" = "npu" ]; then \
        /opt/triton-venv/bin/python -c "import rebel; print('rebel-compiler', rebel.__version__); from rebel.triton.backends import backends; ks=sorted(backends); print('rebel.triton backends:', ks); assert any(k.lower()=='rebel' for k in ks), 'rebel Triton backend not registered in the image'; print('OK: rebel Triton backend present')"; \
    elif [ "$BUILD_MODE" = "cpu" ]; then \
        TRITON_CPU_BACKEND=1 /opt/triton-venv/bin/python -c "import triton; from triton.backends import backends; ks=sorted(backends); print('Triton backends:', ks); assert 'cpu' in ks, 'CPU Triton backend not registered in the image'; print('OK: CPU Triton backend present')"; \
    fi

# Set default environment
ENV VIRTUAL_ENV="/opt/triton-venv"
ENV PATH="/opt/triton-venv/bin:$PATH"

# Expose port for Jupyter if needed
EXPOSE 8888

# Default command - run tests (will auto-detect device)
CMD ["/opt/triton-venv/bin/python", "triton_test.py", "--device", "auto"]
