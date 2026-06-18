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

### G1 CLAMP2 Deployment

For G1 CLAMP2-style policies exported to ONNX, use `g1_clamp2.py`. It reads the embedded deployment metadata, reconstructs the exported observation layout, supports the CLAMP2 `history` observation term, and rebuilds the motion command from a reference `.npz`.

Example:

```bash
ros2 run huro g1_clamp2.py \
  --onnx-path src/huro/resources/policies/g1/g1_clamp2.onnx \
  --motion-npz src/huro/resources/motions/accad_General_A6___Lift_Box.npz 
```

The app supports `JointRefAnchorRpMotionCommand` policies, using a command payload of joint position, joint velocity, root linear velocity in base x/y, root yaw rate, root height, root roll, and root pitch.
The app starts from a fixed default reference: ONNX `default_joint_pos`, zero joint/base velocity, zero roll/pitch, and height `0.78 m`. Press `Space` or `Enter` to transition into the motion reference. When the clip ends, the app transitions back to the fixed default reference and waits for another `Space` or `Enter` press. Press `Space` or `Enter` during playback to transition back early. Press `x` in the terminal to disable the motors.

To choose from every `.npz` clip under `resources/motions`, use the motion imitation runner:

```bash
ros2 run huro g1_motion_imitation.py \
  --onnx-path src/huro/resources/policies/g1/g1_clamp2.onnx
```

The runner starts from the same fixed default reference, prints the available motion list, and waits. Type a motion number then `Enter` to transition into that clip. When the clip ends, it transitions back to the default reference and waits for another selection. Press `Space` or empty `Enter` to replay the selected clip, `l` to reprint the list, `r` to rescan the motions folder, and `x` to disable the motors.
