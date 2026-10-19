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

PTXAS_WRAPPER=false
if [ "$GPU" = true ]; then
    IMAGE_NAME="social-jym:gpu"
    JAX_VARIANT="gpu"
    # GPUs newer than Hopper (compute capability >= 10, e.g. Blackwell RTX 50xx) are unknown to jax 0.4.30:
    # they need a separate image with a ptxas wrapper (see docker/ptxas_wrapper.sh)
    GPU_CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | cut -d. -f1 | sort -n | tail -n 1)
    if [ -n "$GPU_CC_MAJOR" ] && [ "$GPU_CC_MAJOR" -ge 10 ]; then
        PTXAS_WRAPPER=true
        IMAGE_NAME="social-jym:gpu-blackwell"
        echo "Detected a GPU with compute capability $GPU_CC_MAJOR.x: using the image with the ptxas wrapper"
    fi
else
    IMAGE_NAME="social-jym:cpu"
    JAX_VARIANT="cpu"
fi

# Build if requested or if image is not present
if [ "$BUILD" = true ] || ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Building $IMAGE_NAME..."
    docker build \
        --build-arg JAX_VARIANT="$JAX_VARIANT" \
        --build-arg PTXAS_WRAPPER="$PTXAS_WRAPPER" \
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
    # Use nvidia-container-toolkit when Docker can (--gpus, or its CDI devices), otherwise pass the GPU by hand
    GPU_ARGS=()
    for candidate in "--gpus all" "--device nvidia.com/gpu=all"; do
        if docker run --rm $candidate --entrypoint true "$IMAGE_NAME" > /dev/null 2>&1; then
            read -ra GPU_ARGS <<< "$candidate"
            break
        fi
    done
    if [ ${#GPU_ARGS[@]} -gt 0 ]; then
        DOCKER_ARGS+=("${GPU_ARGS[@]}")
    else
        echo "Note: Docker cannot attach the GPU (nvidia-container-toolkit missing, or Docker not restarted after installing it): passing it manually"
        # Same as nvidia-container-toolkit would do: device nodes + host driver libraries
        for dev in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
            [ -e "$dev" ] && DOCKER_ARGS+=(--device "$dev")
        done
        LDCONFIG=$(command -v ldconfig || echo /sbin/ldconfig)
        for lib in libcuda.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4 libnvidia-ml.so.1; do
            lib_path=$("$LDCONFIG" -p | awk -v l="$lib" '$1 == l && /x86-64/ {print $NF; exit}')
            [ -n "$lib_path" ] && DOCKER_ARGS+=(-v "$(readlink -f "$lib_path")":/usr/local/nvidia/lib64/"$lib":ro)
        done
        NVIDIA_SMI=$(command -v nvidia-smi) && DOCKER_ARGS+=(-v "$NVIDIA_SMI":/usr/local/bin/nvidia-smi:ro)
        DOCKER_ARGS+=(-e LD_LIBRARY_PATH=/usr/local/nvidia/lib64)
    fi
fi

echo "=========================================================="
echo "Starting $IMAGE_NAME container"
echo "Discovery Server IP:   $TB4_IP"
echo "Discovery Port:        $TB4_PORT"
echo "ROS Domain ID:         $TB4_DOMAIN_ID"
echo "=========================================================="

docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" bash
