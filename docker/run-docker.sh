#!/bin/bash

# Triton Docker Run Script
# Runs various Triton operations in Docker containers

set -e

DOCKER_IMAGE_REF="${DOCKER_IMAGE_LOCAL:-triton-local-build}:${DOCKER_IMAGE_TAG:-latest}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  test        - Run Triton tests with local build"
    echo "  test-cpu    - Run tests on CPU only"
    echo "  test-cuda   - Run tests on CUDA"
    echo "  test-npu    - Run tests on Rebellions NPU"
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

# Check if the selected image exists.
check_image() {
    if ! docker image inspect "$DOCKER_IMAGE_REF" > /dev/null 2>&1; then
        echo "❌ Docker image '$DOCKER_IMAGE_REF' not found."
        echo "Please build the matching device image first."
        exit 1
    fi
}

sync_test_sources() {
    echo ">>> Copying current Triton test sources to container..."
    docker cp triton_test.py "$CONTAINER_NAME:/workspace/triton_test.py"
    docker exec "$CONTAINER_NAME" rm -rf /workspace/triton_tests
    docker cp triton_tests "$CONTAINER_NAME:/workspace/triton_tests"
}

sync_npu_examples() {
    echo ">>> Copying current NPU integration examples to container..."
    docker exec "$CONTAINER_NAME" rm -rf /workspace/tests
    docker cp tests "$CONTAINER_NAME:/workspace/tests"
}

# Run tests
run_test() {
    local device=${1:-"auto"}
    echo "🧪 Running Triton tests with local build (device: $device)..."

    if [ "$device" = "auto" ]; then
        case "${DOCKER_IMAGE_TAG:-}" in
            npu-latest)
                device="npu"
                ;;
            gpu-latest)
                device="cuda"
                ;;
            cpu-latest)
                device="cpu"
                ;;
            *)
                if docker info | grep -q "nvidia"; then
                    device="cuda"
                else
                    device="cpu"
                fi
                ;;
        esac
    fi
    
    
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
    
    if [ "$device" = "npu" ]; then
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
            "$DOCKER_IMAGE_REF" \
            sleep 300

        sync_test_sources
        sync_npu_examples

        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            env -u TRITON_BACKENDS_IN_TREE \
            /opt/triton-venv/bin/python triton_test.py --device npu

    elif [ "$device" = "cuda" ]; then
        echo "🚀 Using NVIDIA runtime with CUDA..."

        # Start container in background
        docker run -d --name "$CONTAINER_NAME" \
            --runtime=nvidia \
            --gpus all \
            -w /workspace \
            -e TRITON_BACKENDS_IN_TREE=1 \
            -e PYTHONPATH="" \
            "$DOCKER_IMAGE_REF" \
            sleep 300 

        sync_test_sources

        # Execute test
        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            /opt/triton-venv/bin/python triton_test.py --local-triton --device cuda

    else
        echo "🖥️  NVIDIA runtime not available, running without GPU support..."

        # Start container in background
        docker run -d --name "$CONTAINER_NAME" \
            -w /workspace \
            -e TRITON_CPU_BACKEND=1 \
            -e PYTHONPATH="" \
            "$DOCKER_IMAGE_REF" \
            sleep 300

        sync_test_sources

        # Execute test
        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            env TRITON_CPU_BACKEND=1 \
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
    
    
    # Check if NVIDIA runtime is available
    if docker info | grep -q "nvidia"; then
        echo "🚀 Using NVIDIA runtime..."
        docker run -it --rm \
            --runtime=nvidia \
            --gpus all \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            -e TRITON_BACKENDS_IN_TREE=1 \
            "$DOCKER_IMAGE_REF" \
            /opt/triton-venv/bin/python triton_test.py --local-triton --detailed
    else
        echo "🖥️  NVIDIA runtime not available, running CPU-only tests..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            -e TRITON_BACKENDS_IN_TREE=1 \
            "$DOCKER_IMAGE_REF" \
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
            "$DOCKER_IMAGE_REF" \
            bash
    else
        echo "🖥️  NVIDIA runtime not available, starting without GPU support..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            "$DOCKER_IMAGE_REF" \
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
            "$DOCKER_IMAGE_REF" \
            bash -c "/opt/triton-venv/bin/pip install jupyter && /opt/triton-venv/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    else
        echo "🖥️  NVIDIA runtime not available, starting without GPU support..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            -p 8888:8888 \
            "$DOCKER_IMAGE_REF" \
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
            "$DOCKER_IMAGE_REF" \
            bash -c "source /opt/triton-venv/bin/activate && bash"
    else
        echo "🖥️  NVIDIA runtime not available, starting without GPU support..."
        docker run -it --rm \
            -v "$(pwd)":/workspace/low_api_test \
            -w /workspace \
            "$DOCKER_IMAGE_REF" \
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
    docker logs $(docker ps -q --filter "ancestor=$DOCKER_IMAGE_REF") 2>/dev/null || echo "No running containers found."
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
