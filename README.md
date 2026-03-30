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

### G1 Motion Tracking Deployment

For G1 motion-tracking policies exported to ONNX, use the dedicated `g1_motion_tracking.py` deploy app. It reads the embedded ONNX deployment metadata, reconstructs the policy observation history online, and tracks a reference motion clip from an `.npz` file.

Example:

```bash
ros2 run huro g1_motion_tracking.py \
  --onnx-path src/huro/resources/policies/g1/g1_tracking.onnx \
  --motion-npz /path/to/reference_motion.npz
```

Some example motions are available inside `src/huro/resources/motions/accad_subset/`.

The app starts in default-reference mode and keeps running the policy, using a frozen standing pose as the motion-command reference.
Press `Space` or `Enter` in the terminal to start the motion clip from the beginning, with a smooth transition into the clip start. When the clip finishes, the command automatically transitions back to the standing reference pose and waits for the next key press. Press `x` in the terminal to disable the motors.

### G1 Hand-Base Deployment

For G1 Hand-Base policies exported to ONNX, use the dedicated `g1_hand_base.py` deploy app. It reads the embedded ONNX deployment metadata, loads a motion `.npz`, and reconstructs the 22-D Hand-Base command online from the clip's pelvis and wrist poses.

Example:

```bash
ros2 run huro g1_hand_base.py \
  --onnx-path /path/to/g1_hand_base.onnx \
  --motion-npz /path/to/reference_motion.npz
```

This first version uses the motion clip itself as the Hand-Base command source. Later it can be extended to accept direct base velocity and hand pose targets instead of an `.npz`.

### G1 Hand-Base Interactive Control

For direct keyboard + RViz control of a G1 Hand-Base policy, use the split-node setup:

```bash
ros2 run huro g1_hand_base_policy.py \
  --onnx-path /path/to/g1_hand_base.onnx
```

In a second terminal:

```bash
ros2 run huro g1_hand_base_teleop.py
```

When running inside Docker, start the teleop node from an interactive shell so it has a real TTY for keyboard input. If you attach with `docker exec`, make sure to use `-it`.

The policy node runs the Hand-Base policy directly, using hard-coded default command values until the teleop topics overwrite them: zero base velocity, a fixed default height, and fixed default left/right hand references. The ONNX export is expected to use `joint_position` action semantics. The teleop node publishes:

- `/g1_hand_base/cmd_vel` from keyboard input for planar base velocity and yaw rate
- `/g1_hand_base/height` from keyboard input for base height
- `/g1_hand_base/left_hand_target` and `/g1_hand_base/right_hand_target` from RViz interactive markers

The interactive markers are initialized from the live TF pose of `left_wrist_yaw_link` and `right_wrist_yaw_link`, so this setup expects `root_g1` and `robot_state_publisher` to already be running.
RViz and the teleop node must also share the same ROS graph as the policy node, whether they are running in the same container or across host/container terminals.
