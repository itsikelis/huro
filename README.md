# HURo: HuCeBot Unitree Robot Interface

## Installation

### Docker

To build the docker image, navigate to the repository root folder and type
```bash
cd docker && ./build.sh
```

To launch the docker container (first or recurring instances) execute
```bash
cd docker && ./run.sh
```

## Usage

At the moment, HURo has been tested on the Unitree G1 humanoid and Unitree Go2 quadruped robots. It supports deployment both on hardware and on a MuJoCo simulation.

### Workspace preparation

To build the package code, from an interactive container session run:

```bash
colcon build
```

Then, set up the CycloneDDS network interface by running:

```bash
source setup_uri.sh INTERFACE_NAME
```

Replace INTERFACE_NAME with the name of the network interface (Ethernet, WiFi or lo). To check the available network interfaces you can run:

```bash
ip a
```

**Important note**: To run a simulation node, you should use lo (the loopback address).

**Important note**: If you wish to execute things over Ethernet, you need to set up a wired connection with a static IP in Linux. Set the following in NetworkManager:

```
IP address: 192.168.123.222
Netmask: 24
Gateway: 192.168.123.1
```

### Run the root node

The root node acts as an intermediary between the custom Unitree message types and the standard ROS2 messages used by RViz.

In the future, the Root node will also act as a safety layer to monitor joint or effort limit violation and motor temperatures.

To run the Root node, spawn a container interactive session and run:

```bash
ros2 launch huro ROBOT_rviz.launch.py
```

replacing robot with either "g1" or "go2".

If you are connected to a robot, this will open up an RViz window that updates joint and floating base position as these move.

### Run the simulation node

If you wish to see a simulated robot, open a different container interactive session and run:

```bash
ros2 run huro sim_ROBOT
```

replacing robot with either "g1" or "go2".
