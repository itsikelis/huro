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

The runner starts from the same fixed default reference, prints the available motion list, and waits. Type a motion number then `Enter` to transition into that clip. When the clip ends, it transitions back to the default reference and waits for another selection. Press `Space` or empty `Enter` to replay the selected clip, `s` to pause/resume motion playback with the reference held constant, `l` to reprint the list, `r` to rescan the motions folder, and `x` to disable the motors.

By default in Docker, logs are saved under `/huro_ws/src/huro/resources/log/g1_motion_imitation/<run_timestamp>/`. Override the parent folder with `--log-dir ...` if needed, or add `--log-label my_test` to use `<run_timestamp>_my_test/`. Each `.npz` file contains transition-in, motion, optional paused-motion, and transition-out samples, with `phase_id` values matching `meta_phase_names`.

For hardcoded stationary stance references, use:

```bash
ros2 run huro g1_predefined_stance.py \
  --onnx-path src/huro/resources/policies/g1/g1_clamp2.onnx \
  --pose bent_forearms
```

Available poses are `bent_forearms` and `arms_forward`. The runner transitions from the ONNX default stance into the selected pose at startup. Press `1` or `2` to switch pose, `Space` or `Enter` to toggle default/selected stance, and `x` to disable the motors.

By default in Docker, logs are saved under `/huro_ws/src/huro/resources/log/g1_predefined_stance/<run_timestamp>/`. Override the parent folder with `--log-dir ...` if needed, or add `--log-label my_test` to use `<run_timestamp>_my_test/`. Each `.npz` file contains transition-in, hold, and transition-out samples, with `phase_id` values matching `meta_phase_names`.

To compare logged real robot state against the reference and desired commands:

```bash
python3 scripts/plot_deploy_npz.py resources/log/g1_motion_imitation/2026-06-19_07-53-06
```

The script writes PNG overlays and RMSE summaries to a `plots/` folder next to each `.npz` by default. New logs also include low-level actuator fields such as `joint_torques_hw`, `joint_accelerations_hw`, `motor_temperature_ch0_hw`, `motor_temperature_ch1_hw`, `motor_voltage_hw`, `motor_state_flags_hw`, and `motor_mode_hw`.
