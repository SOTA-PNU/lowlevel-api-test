#!/bin/bash

set -e

DOCKER_IMAGE_REF="${DOCKER_IMAGE:-ghcr.io/sota-pnu/lowlevel-api-test}:${DOCKER_IMAGE_TAG}"
COMMANDS="test-cpu test-cuda test-npu"
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

check_image() {
    if [ -z "${DOCKER_IMAGE_TAG:-}" ]; then
        echo "DOCKER_IMAGE_TAG is not set. Use cpu-latest, gpu-latest, or npu-latest."
        exit 1
    fi

    if ! docker image inspect "$DOCKER_IMAGE_REF" > /dev/null 2>&1; then
        echo "Docker image '$DOCKER_IMAGE_REF' not found."
        echo "Please build the matching device image first."
        exit 1
    fi
}

sync_test_sources() {
    echo ">>> Copying current Triton test sources to container..."

    local source
    for source in triton_test.py cpu_gpu.py npu.py results.py benchmark.py; do
        docker cp "$source" "$CONTAINER_NAME:/workspace/$source"
    done
}

run_test() {
    local device="$1"
    echo "Running Triton tests on device: $device"
    CONTAINER_NAME="triton-test-$$"
    
    cleanup_container() {
        if [ -n "$CONTAINER_NAME" ]; then
            echo ">>> Cleaning up container: $CONTAINER_NAME"
            docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
        fi
    }
    trap cleanup_container EXIT

    echo ">>> Creating temporary container: $CONTAINER_NAME"
    
    if [ "$device" = "npu" ]; then
        echo "Starting NPU test container..."

        docker run -d --name "$CONTAINER_NAME" \
            --device rebellions.ai/npu=all \
            --ipc=host \
            -w /workspace \
            -e PYTHONPATH="" \
            "$DOCKER_IMAGE_REF" \
            sleep infinity

        sync_test_sources

        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            env -u TRITON_BACKENDS_IN_TREE \
            /opt/triton-venv/bin/python -u triton_test.py --device npu

    elif [ "$device" = "cuda" ]; then
        echo "Starting GPU test container..."

        docker run -d --name "$CONTAINER_NAME" \
            --runtime=nvidia \
            --gpus all \
            -w /workspace \
            -e PYTHONPATH="" \
            "$DOCKER_IMAGE_REF" \
            sleep infinity 

        sync_test_sources

        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            /opt/triton-venv/bin/python -u triton_test.py --device cuda

    elif [ "$device" = "cpu" ]; then
        echo "Starting CPU test container..."

        docker run -d --name "$CONTAINER_NAME" \
            -w /workspace \
            -e TRITON_CPU_BACKEND=1 \
            -e PYTHONPATH="" \
            "$DOCKER_IMAGE_REF" \
            sleep infinity

        sync_test_sources

        echo ">>> Running tests..."
        docker exec "$CONTAINER_NAME" \
            /opt/triton-venv/bin/python -u triton_test.py --device cpu

    else
        echo "Unknown device: $device"
        exit 1
    fi
}

main() {
    check_docker
    
    case "$1" in
        "test-cpu")
            check_image
            run_test "cpu"
            ;;
        "test-cuda")
            check_image
            run_test "cuda"
            ;;
        "test-npu")
            check_image
            run_test "npu"
            ;;
        "help"|"-h"|"--help")
            echo "Commands: $COMMANDS"
            ;;
        "")
            echo "No command specified."
            echo "Commands: $COMMANDS"
            exit 1
            ;;
        *)
            echo "Unknown command: $1"
            echo "Commands: $COMMANDS"
            exit 1
            ;;
    esac
}

main "$@"