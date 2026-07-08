#!/bin/bash

# Triton Docker Run Script
# Runs various Triton operations in Docker containers

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  test        - Run Triton tests with local build"
    echo "  test-cpu    - Run tests on CPU only"
    echo "  test-cuda   - Run tests on CUDA"
    echo "  test-detailed - Run detailed functional tests"
    echo "  dev         - Start development container with interactive shell"
    echo "  jupyter     - Start Jupyter Lab server"
    echo "  bash        - Start interactive bash session"
    echo "  clean       - Clean up Docker containers and images"
    echo "  logs        - Show container logs"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 dev"
    echo "  $0 jupyter"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Check if image exists
check_image() {
    if ! docker image inspect triton-local-build:latest > /dev/null 2>&1; then
        echo "❌ Docker image 'triton-local-build:latest' not found."
        echo "Please run './build-docker.sh' first to build the image."
        exit 1
    fi
}

# Check and build Triton if needed
ensure_triton_build() {
    echo "🔍 Checking Triton build status..."
    
    # Test if Triton can be imported in the container (using the pre-built version in /opt/triton-src)
    if docker run --rm triton-local-build:latest /opt/triton-venv/bin/python -c "import triton; print('Triton is ready')" > /dev/null 2>&1; then
        echo "✅ Triton is pre-built in the Docker image"
    else
        echo "⚠️  Triton not found or corrupted in Docker image"
        echo "🔄 Building Triton inside container..."
        
        # Build Triton inside the container using the /opt/triton-src location
        docker run --rm \
            -v "$(pwd)":/workspace \
            -w /workspace \
            triton-local-build:latest \
            bash -c "
                echo '🔧 Setting up Triton build environment...'
                cd /opt/triton-src
                
                # Detect GPU architecture if CUDA is available
                if command -v nvidia-smi &> /dev/null; then
                    GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
                    echo \"🎮 Detected GPU: \$GPU_NAME\"
                    
                    case \"\$GPU_NAME\" in
                        *\"Blackwell\"*|*\"RTX PRO 6000\"*)
                            CUDA_ARCH=\"sm_120\"
                            echo \"🏗️  Using CUDA architecture: \$CUDA_ARCH (Blackwell)\"
                            ;;
                        *\"Hopper\"*|*\"H100\"*)
                            CUDA_ARCH=\"sm_90\"
                            echo \"🏗️  Using CUDA architecture: \$CUDA_ARCH (Hopper)\"
                            ;;
                        *\"Ampere\"*|*\"A100\"*|*\"RTX 30\"*|*\"RTX 40\"*)
                            CUDA_ARCH=\"sm_80\"
                            echo \"🏗️  Using CUDA architecture: \$CUDA_ARCH (Ampere)\"
                            ;;
                        *)
                            CUDA_ARCH=\"sm_80\"
                            echo \"🏗️  Unknown GPU, using default CUDA architecture: \$CUDA_ARCH\"
                            ;;
                    esac
                    
                    export TRITON_CUDA_ARCHITECTURES=\"\$CUDA_ARCH\"
                    export TORCH_CUDA_ARCH_LIST=\"\$CUDA_ARCH\"
                    echo \"🚀 Building Triton with CUDA architecture: \$CUDA_ARCH\"
                else
                    echo \"🖥️  No GPU detected, building for CPU-only\"
                fi
                
                # Install Triton in development mode
                echo '📦 Installing Triton in development mode...'
                /opt/triton-venv/bin/pip install -e .
                
                # Build extensions
                echo '🔨 Building Triton extensions...'
                /opt/triton-venv/bin/python setup.py build_ext --inplace
                
                echo '✅ Triton build completed successfully!'
            "
        
        echo "✅ Triton build completed"
    fi
}

# Run tests
# run_test() {
#     local device=${1:-"auto"}
#     echo "🧪 Running Triton tests with local build (device: $device)..."
    
#     # Ensure Triton is built
#     ensure_triton_build
    
#     # Check if NVIDIA runtime is available
#     if docker info | grep -q "nvidia"; then
#         echo "🚀 Using NVIDIA runtime with CUDA..."
#         docker run --rm \
#             --runtime=nvidia \
#             --gpus all \
#             -v "$(pwd)":/workspace/low_api_test \
#             -w /workspace \
#             -e TRITON_BACKENDS_IN_TREE=1 \
#             triton-local-build:latest \
#             /opt/triton-venv/bin/python triton_test.py --local-triton --device cuda
#     else
#         echo "🖥️  NVIDIA runtime not available, running without GPU support..."
#         docker run --rm \
#             -v "$(pwd)":/workspace/low_api_test \
#             -w /workspace \
#             -e TRITON_BACKENDS_IN_TREE=1 \
#             triton-local-build:latest \
#             /opt/triton-venv/bin/python triton_test.py --local-triton --device cpu
#     fi
# }
run_test() {
    local device=${1:-"auto"}
    echo "🧪 Running Triton tests with local build (device: $device)..."
    
    # Ensure Triton is built
    ensure_triton_build
    
    # Generate unique container name
    CONTAINER_NAME="triton-test-$$-$(date +%s)"
    
    # Setup cleanup trap to ensure container is always removed
    cleanup_container() {
        if [ -n "$CONTAINER_NAME" ]; then
            echo ">>> Cleaning up container: $CONTAINER_NAME"
            docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
        fi
    }
    trap cleanup_container EXIT
    
    echo ">>> Creating temporary container: $CONTAINER_NAME"
    
    if [ "$device" = "npu" ] || [ "${DOCKER_IMAGE_TAG:-}" = "npu-latest" ]; then
        echo "🧠 Using NPU container runtime access..."

        # NPU device nodes and vendor runtime libraries are environment-specific.
        # --privileged lets the self-hosted NPU runner expose its device stack to
        # the container; NPU_DOCKER_ARGS can add site-specific mounts if needed.
        docker run -d --name "$CONTAINER_NAME" \
            --privileged \
            --ipc=host \
            ${NPU_DOCKER_ARGS:-} \
            -w /workspace \
            -e PYTHONPATH="" \
            triton-local-build:latest \
            sleep 300

        echo ">>> Copying triton_test.py to container..."
        docker cp triton_test.py "$CONTAINER_NAME:/workspace/triton_test.py"

        echo ">>> Copying tests to container..."
        docker cp tests "$CONTAINER_NAME:/workspace/tests"

        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            env -u TRITON_BACKENDS_IN_TREE \
            /opt/triton-venv/bin/python triton_test.py --device npu

    # CPU images should skip CUDA-only execution tests. Other images keep
    # the existing NVIDIA runtime detection behavior.
    elif [ "$device" != "cpu" ] && [ "${DOCKER_IMAGE_TAG:-}" != "cpu-latest" ] && docker info | grep -q "nvidia"; then
        echo "🚀 Using NVIDIA runtime with CUDA..."

        # Start container in background
        docker run -d --name "$CONTAINER_NAME" \
            --runtime=nvidia \
            --gpus all \
            -w /workspace \
            -e TRITON_BACKENDS_IN_TREE=1 \
            -e PYTHONPATH="" \
            triton-local-build:latest \
            sleep 300

        # Copy triton_test.py into container
        echo ">>> Copying triton_test.py to container..."
        docker cp triton_test.py "$CONTAINER_NAME:/workspace/triton_test.py"

        # Execute test
        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            /opt/triton-venv/bin/python triton_test.py --local-triton --device cuda

    else
        echo "🖥️  NVIDIA runtime not available, running without GPU support..."

        # Start container in background
        docker run -d --name "$CONTAINER_NAME" \
            -w /workspace \
            -e TRITON_BACKENDS_IN_TREE=1 \
            -e PYTHONPATH="" \
            triton-local-build:latest \
            sleep 300

        # Copy triton_test.py into container
        echo ">>> Copying triton_test.py to container..."
        docker cp triton_test.py "$CONTAINER_NAME:/workspace/triton_test.py"

        # Execute test
        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            /opt/triton-venv/bin/python triton_test.py --local-triton --device cpu
    fi

    # Cleanup will be done automatically by trap on function exit
}

# Run CPU-only tests
run_test_cpu() {
    echo "🖥️  Running CPU-only tests..."
    run_test "cpu"
}

# Run CUDA tests
run_test_cuda() {
    echo "🚀 Running CUDA tests..."
    run_test "cuda"
}

# Run NPU tests
run_test_npu() {
    echo "🧠 Running NPU tests..."
    run_test "npu"
}

# Run detailed tests
run_test_detailed() {
    echo "🔍 Running detailed tests..."
    
    # Ensure Triton is built
    ensure_triton_build
    
    # Check if NVIDIA runtime is available
    if docker info | grep -q "nvidia"; then
        echo "🚀 Using NVIDIA runtime..."
        docker run -it --rm \
            --runtime=nvidia \
            --gpus all \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            -e TRITON_BACKENDS_IN_TREE=1 \
            triton-local-build:latest \
            /opt/triton-venv/bin/python triton_test.py --local-triton --detailed
    else
        echo "🖥️  NVIDIA runtime not available, running CPU-only tests..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            -e TRITON_BACKENDS_IN_TREE=1 \
            triton-local-build:latest \
            /opt/triton-venv/bin/python triton_test.py --local-triton --device cpu
    fi
}

# Run development container
run_dev() {
    echo "🚀 Starting development container..."
    
    # Check if NVIDIA runtime is available
    if docker info | grep -q "nvidia"; then
        echo "🚀 Using NVIDIA runtime..."
        docker run -it --rm \
            --runtime=nvidia \
            --gpus all \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            triton-local-build:latest \
            bash
    else
        echo "🖥️  NVIDIA runtime not available, starting without GPU support..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            triton-local-build:latest \
            bash
    fi
}

# Start Jupyter Lab
run_jupyter() {
    echo "📓 Starting Jupyter Lab..."
    
    # Check if NVIDIA runtime is available
    if docker info | grep -q "nvidia"; then
        echo "🚀 Using NVIDIA runtime..."
        docker run -it --rm \
            --runtime=nvidia \
            --gpus all \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            -p 8888:8888 \
            triton-local-build:latest \
            bash -c "/opt/triton-venv/bin/pip install jupyter && /opt/triton-venv/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    else
        echo "🖥️  NVIDIA runtime not available, starting without GPU support..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            -p 8888:8888 \
            triton-local-build:latest \
            bash -c "/opt/triton-venv/bin/pip install jupyter && /opt/triton-venv/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    fi
}

# Start interactive bash
run_bash() {
    echo "🐚 Starting interactive bash session..."
    
    # Check if NVIDIA runtime is available
    if docker info | grep -q "nvidia"; then
        echo "🚀 Using NVIDIA runtime..."
        docker run -it --rm \
            --runtime=nvidia \
            --gpus all \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            triton-local-build:latest \
            bash -c "source /opt/triton-venv/bin/activate && bash"
    else
        echo "🖥️  NVIDIA runtime not available, starting without GPU support..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            triton-local-build:latest \
            bash -c "source /opt/triton-venv/bin/activate && bash"
    fi
}

# Clean up Docker resources
clean_docker() {
    echo "🧹 Cleaning up Docker resources..."
    docker container prune -f
    docker image prune -f
    docker system prune -f
    echo "✅ Cleanup completed!"
}

# Show logs
show_logs() {
    echo "📋 Showing container logs..."
    docker logs $(docker ps -q --filter ancestor=triton-local-build:latest) 2>/dev/null || echo "No running containers found."
}

# Main script logic
main() {
    check_docker
    
    case "${1:-}" in
        "test")
            check_image
            run_test
            ;;
        "test-cpu")
            check_image
            run_test_cpu
            ;;
        "test-cuda")
            check_image
            run_test_cuda
            ;;
        "test-npu")
            check_image
            run_test_npu
            ;;
        "test-detailed")
            check_image
            run_test_detailed
            ;;
        "dev")
            check_image
            run_dev
            ;;
        "jupyter")
            check_image
            run_jupyter
            ;;
        "bash")
            check_image
            run_bash
            ;;
        "clean")
            clean_docker
            ;;
        "logs")
            show_logs
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        "")
            echo "❌ No command specified."
            show_usage
            exit 1
            ;;
        *)
            echo "❌ Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
