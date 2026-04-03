#!/bin/bash

CONTAINER_NAME="huro_container"
IMAGE_NAME="huro"

# Auto-select Gazebo render engine.
# Override manually with: export GZ_RENDER_ENGINE=ogre2 (or ogre)
REQUESTED_RENDER_ENGINE="${GZ_RENDER_ENGINE:-auto}"

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1 \
   && nvidia-smi -L >/dev/null 2>&1 \
   && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    GPU_ARGS+=(--gpus all)
    GPU_ARGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
    GPU_ARGS+=(--env NVIDIA_DRIVER_CAPABILITIES=all)
    GPU_AVAILABLE=true
else
    GPU_AVAILABLE=false
fi

if [[ "$REQUESTED_RENDER_ENGINE" == "auto" ]]; then
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        RENDER_ENGINE="ogre2"
    else
        RENDER_ENGINE="ogre"
    fi
else
    RENDER_ENGINE="$REQUESTED_RENDER_ENGINE"
fi

xhost +

# Check if container is already running
if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "Container '$CONTAINER_NAME' is already running. Opening a new terminal..."
    docker exec -it $CONTAINER_NAME bash
else
    echo "Starting a new container named '$CONTAINER_NAME'..."
    echo "Render engine: $RENDER_ENGINE"
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        echo "GPU runtime: NVIDIA detected"
    else
        echo "GPU runtime: not detected (using CPU-compatible settings)"
    fi
    docker run \
        --interactive \
        --tty \
        --rm \
        --network host \
        --env DISPLAY=$DISPLAY \
        --env GZ_RENDER_ENGINE=$RENDER_ENGINE \
        --privileged \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --volume $(pwd)/setup_uri.sh:/huro_ws/setup_uri.sh \
        --volume $(pwd)/../:/huro_ws/src/huro \
        --workdir /huro_ws \
        --name $CONTAINER_NAME \
        "${GPU_ARGS[@]}" \
        $IMAGE_NAME
        # -v $(pwd)/config/livox_mid.json:/huro_ws/src/livox_ros_driver2/config/MID360_config.json  \
fi
