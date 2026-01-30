# HURo: HuCeBot Unitree Robot Interface

## Installation

### Docker

We provide utility scripts to build and run the HURo docker image in the docker folder:
```bash
cd docker
```

To build the docker image:
```bash
cd docker && ./build.sh
```

To launch a docker container (first or subsequent instances):
```bash
./run.sh
```

## Usage

Until now, HURo has been tested on the Unitree G1 humanoid and Unitree Go2 quadruped robots. It supports deployment both on hardware and on a MuJoCo simulation.

### Workspace preparation

To build the workspace code and examples, launch an interactive container session:
```bash
cd docker && ./run.sh
```

Then build with colcon:
```bash
colcon build
```

HURo supports seamless simulation (MuJoCo) and real robot deployment.
This is achieved by selecting the appropriate network interface and setting it up on CycloneDDS.

#### Simulation deployment

In each docker terminal source the setup_uri script and pass lo (loopback) as an argument:
```bash
docker> source setup_uri.sh lo
```

Run the root, simulation and rviz node:
```bash
ros2 launch huro ROBOT_sim.launch.py
```

replacing robot with either "g1" or "go2".


#### Robot deployment (Ethernet only for now)

Set up a wired connection with the following:

```
IP address: 192.168.123.222 (static)
Netmask: 24
Gateway: 192.168.123.1
```

Check the available network interfaces and note down your ethernet interface by running:
```bash
ip a
```

In each docker terminal source the setup_uri script and pass the ethernet interface name as an argument:
```bash
docker> source setup_uri.sh ETH_INTERFACE
```

Run the root, and rviz node:
```bash
ros2 launch huro ROBOT_rviz.launch.py
```
If you are connected to a robot, this will open up an RViz window that updates joint and floating base position as these move.
