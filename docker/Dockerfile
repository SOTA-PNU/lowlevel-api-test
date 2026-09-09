# syntax=docker/dockerfile:1

FROM ubuntu:22.04

ARG BUILD_MODE=cuda
ARG CUDA_VERSION=12.8
ARG PYTORCH_CUDA_VERSION=cu128
ARG RBLN_COMPILER_VERSION=0.11.1.post1
ARG CUDA_TORCH_VERSION_CU128=2.11.0
ARG CUDA_TRITON_VERSION_CU128=3.6.0
ARG NPU_TORCH_VERSION=2.8.0
ARG CPU_TORCH_VERSION=2.11.0
ARG TRITON_CPU_REF=9a3dd8096b3c5b89a6dfeba012221f3fed450eb0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/triton-venv/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

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

WORKDIR /workspace
RUN python3 -m venv /opt/triton-venv
RUN /opt/triton-venv/bin/pip install --upgrade pip setuptools wheel

# Install PyTorch, Triton, and other packages based on the build mode
RUN if [ "$BUILD_MODE" = "cuda" ]; then \
        if [ "$PYTORCH_CUDA_VERSION" = "cu128" ]; then \
            TORCH_VERSION="$CUDA_TORCH_VERSION_CU128"; \
            TRITON_VERSION="$CUDA_TRITON_VERSION_CU128"; \
        else \
            echo "ERROR: Unsupported PYTORCH_CUDA_VERSION=${PYTORCH_CUDA_VERSION}" >&2; \
            exit 1; \
        fi && \
        python -m pip install --only-binary=:all: "torch==${TORCH_VERSION}" \
            --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION}" && \
        python -m pip install --only-binary=:all: \
            "triton==${TRITON_VERSION}" numpy; \

    elif [ "$BUILD_MODE" = "npu" ]; then \
        python -m pip install "torch==${NPU_TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu && \
        python -m pip install numpy; \

    elif [ "$BUILD_MODE" = "cpu" ]; then \
        python -m pip install "torch==${CPU_TORCH_VERSION}" \
            --index-url https://download.pytorch.org/whl/cpu && \
        python -m pip install numpy; \

    else \
        echo "ERROR: Unsupported BUILD_MODE=${BUILD_MODE}" >&2; exit 1; \
    fi

RUN if [ "$BUILD_MODE" = "cpu" ]; then \
        git clone https://github.com/triton-lang/triton-cpu.git /opt/triton-src && \
        cd /opt/triton-src && \
        git checkout "$TRITON_CPU_REF" && \
        git submodule update --init --recursive && \
        python -m pip install -r python/requirements.txt && \
        TRITON_CPU_BACKEND=1 python -m pip install -e . --no-build-isolation; \
    fi

RUN --mount=type=secret,id=rbln_pip_config,required=false \
    if [ "$BUILD_MODE" = "npu" ]; then \
        test -s /run/secrets/rbln_pip_config || \
            { echo "ERROR: NPU build requires BuildKit secret rbln_pip_config" >&2; exit 1; }; \
        PIP_CONFIG_FILE=/run/secrets/rbln_pip_config \
            /opt/triton-venv/bin/pip install --no-cache-dir \
            "rebel-compiler==${RBLN_COMPILER_VERSION}"; \
    fi

# Check the installation and versions of installed packages
RUN python -m pip check && \
    if [ "$BUILD_MODE" = "cuda" ]; then \
        python -c "import triton; import triton.language; print('pip-installed Triton', triton.__version__, triton.__file__)"; \
    elif [ "$BUILD_MODE" = "npu" ]; then \
        python -c "import rebel; print('rebel-compiler', rebel.__version__); from rebel.triton.backends import backends; print('rebel.triton backends:', sorted(backends)); assert 'rebel' in backends, 'rebel Triton backend not registered in the image'"; \
    elif [ "$BUILD_MODE" = "cpu" ]; then \
        python -c "import triton; from triton.backends import backends; print('triton-cpu', triton.__version__); assert 'cpu' in backends, 'CPU Triton backend not registered in the image'"; \
    fi

COPY . /workspace/
ENV VIRTUAL_ENV="/opt/triton-venv"
CMD ["/opt/triton-venv/bin/python", "triton_test.py", "--device", "auto"]