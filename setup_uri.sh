# Copyright Ioannis Tsikelis
#
# Script to set up the network interface for CycloneDDS

#!/bin/bash
if [ $# -eq 0 ]
then
    echo "Usage: source setup.sh [network interface (or lo)]"
    return 1
fi

echo "Sourcing installed packages"
source /opt/ros/${ROS_DISTRO}/setup.bash
source ./install/setup.bash

# For autocompletion to work in terminal
eval "$(register-python-argcomplete3 ros2)"
eval "$(register-python-argcomplete3 colcon)"

echo "Setting up DDS"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General>
                        <Interfaces>
                        <NetworkInterface name="'$1'" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'

echo "Setting up Gazebo resource paths"
HURO_PREFIX=$(ros2 pkg prefix huro 2>/dev/null)
if [ -n "$HURO_PREFIX" ]
then
    HURO_SHARE="$HURO_PREFIX/share"
    export GZ_SIM_RESOURCE_PATH="$HURO_SHARE:${GZ_SIM_RESOURCE_PATH}"
    export IGN_GAZEBO_RESOURCE_PATH="$HURO_SHARE:${IGN_GAZEBO_RESOURCE_PATH}"
    echo "Gazebo resource root: $HURO_SHARE"
fi

if [ "$1" = "lo"  ]
then
    # Need to enable multicast if using localhost
    echo "Enabling multicast"
    ip link set lo multicast on
fi

echo "Done, let's try!"
