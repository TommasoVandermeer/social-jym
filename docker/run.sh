#!/bin/bash
# Usage:
#   ./docker/run.sh           # CPU-only container
#   ./docker/run.sh --gpu     # NVIDIA GPU container
#   ./docker/run.sh --no-gpu  # explicitly CPU-only

set -e

GPU=false
for arg in "$@"; do
    case $arg in
        --gpu)    GPU=true  ;;
        --no-gpu) GPU=false ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--gpu|--no-gpu]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ "$GPU" = true ]; then
    IMAGE_NAME="social-jym:gpu"
    JAX_VARIANT="gpu"
else
    IMAGE_NAME="social-jym:cpu"
    JAX_VARIANT="cpu"
fi

# Build the image if not already present
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Image $IMAGE_NAME not found, building..."
    docker build \
        --build-arg JAX_VARIANT="$JAX_VARIANT" \
        -t "$IMAGE_NAME" \
        -f "$SCRIPT_DIR/Dockerfile" \
        "$REPO_ROOT"
fi

# Allow X11 connections from local Docker containers
xhost +local:docker > /dev/null 2>&1 || echo "Warning: xhost failed, GUI may not work"

DOCKER_ARGS=(
    -it --rm
    --name social-jym
    -e DISPLAY="$DISPLAY"
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -v "$REPO_ROOT":/opt/social-jym
    --workdir /opt/social-jym
    --network host
)

if [ "$GPU" = true ]; then
    DOCKER_ARGS+=(--gpus all)
fi

echo "Starting $IMAGE_NAME container..."
docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" bash
