#!/bin/bash
# Usage:
#   ./docker/run.sh                    # CPU-only container
#   ./docker/run.sh --gpu              # NVIDIA GPU container
#   ./docker/run.sh --no-gpu           # explicitly CPU-only
#   ./docker/run.sh --build            # rebuild image before running
#   ./docker/run.sh --gpu --build      # rebuild GPU image and run
#   ./docker/run.sh --no-mount         # skip bind mount (needed on NFS home dirs)

set -e

GPU=false
BUILD=false
MOUNT=true
TB4_IP="192.168.8.4"
TB4_PORT="11811"
TB4_DOMAIN_ID="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)      GPU=true; shift ;;
        --no-gpu)   GPU=false; shift ;;
        --build)    BUILD=true; shift ;;
        --no-mount) MOUNT=false; shift ;;
        --ip)       TB4_IP="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--gpu|--no-gpu] [--build] [--no-mount] [--ip <IP>] "
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

# Build if requested or if image is not present
if [ "$BUILD" = true ] || ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Building $IMAGE_NAME..."
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
    --workdir /opt/social-jym
    --network host
    -e RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
    -e ROS_DOMAIN_ID="${TB4_DOMAIN_ID}"
    -e ROS_DISCOVERY_SERVER="${TB4_IP}:${TB4_PORT};"
    -e ROS_SUPER_CLIENT="True"
)

if [ "$MOUNT" = true ]; then
    DOCKER_ARGS+=(-v "$REPO_ROOT":/opt/social-jym)
else
    echo "Note: --no-mount active, using code baked into the image"
fi

if [ "$GPU" = true ]; then
    DOCKER_ARGS+=(--gpus all)
fi

echo "=========================================================="
echo "Starting $IMAGE_NAME container"
echo "Discovery Server IP:   $TB4_IP"
echo "Discovery Port:        $TB4_PORT"
echo "ROS Domain ID:         $TB4_DOMAIN_ID"
echo "=========================================================="

docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" bash
