# HURo: HuCeBot Unitree Robot Interface

RL policy trained with lidar observations deployed with NAV2 stack.

<img src="/docs/gazebo_lidar_rl_compressed.gif" alt="Gazebo lidar RL demo" width="800" />


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

### This branch focuses on the go2 integration with a lidar

### Example to start the go2 policy in Mujoco simulation:

```bash
# first terminal
colcon build
source setup_uri.sh lo
ros2 launch huro go2_rviz.launch.py
ros2 run huro sim_go2
# second terminal
colcon build
source setup_uri.sh lo
ros2 run huro go2_publisher.py --sim True --vx 0.75 --vy 0.0 --wz 0.3
```

You can plug in a controller to control the robot.

### Example to start the go2 policy in Gazebo simulation:

```bash
# first terminal
colcon build
source setup_uri.sh lo
ros2 launch huro go2_sim_gz.launch.py step_height:=0.1
# second terminal
colcon build
source setup_uri.sh lo
ros2 run huro go2_publisher.py --sim True
```

Finally, you can send a goal in rviz with the **Nav2 goal** tool.


### With the real robot:

Comming Soon...