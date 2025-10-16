#!/bin/bash

CONTAINER_NAME="huro_container"
IMAGE_NAME="huro"

xhost +

# Check if container is already running
if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "Container '$CONTAINER_NAME' is already running. Opening a new terminal..."
    docker exec -it $CONTAINER_NAME bash
else
    echo "Starting a new container named '$CONTAINER_NAME'..."
    docker run \
        --interactive \
        --tty \
        --rm \
        --network host \
        --env DISPLAY=$DISPLAY \
        --privileged \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --volume $(pwd)/setup_uri.sh:/huro_ws/setup_uri.sh \
        --volume $(pwd)/../:/huro_ws/src/huro \
        --workdir /huro_ws \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
        # -v $(pwd)/config/livox_mid.json:/huro_ws/src/livox_ros_driver2/config/MID360_config.json  \
fi
