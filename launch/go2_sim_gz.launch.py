import os
import tempfile
import yaml
import xml.etree.ElementTree as ET

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory


DEFAULT_STEP_HEIGHT = 0.06


def _scale_stairs_world(world_template_path: str, step_height: float) -> str:
    """Create a temporary world file with staircase Z dimensions scaled from step height."""
    tree = ET.parse(world_template_path)
    root = tree.getroot()
    stairs_model = root.find(".//model[@name='stairs_up_down']")
    if stairs_model is None:
        return world_template_path

    scale = step_height / DEFAULT_STEP_HEIGHT

    for elem in list(stairs_model.findall(".//collision")) + list(stairs_model.findall(".//visual")):
        name = elem.get("name", "")
        if not name.startswith("step_"):
            continue

        pose_elem = elem.find("pose")
        if pose_elem is not None and pose_elem.text:
            pose_vals = pose_elem.text.split()
            if len(pose_vals) >= 3:
                pose_vals[2] = f"{float(pose_vals[2]) * scale:.6f}".rstrip("0").rstrip(".")
                pose_elem.text = " ".join(pose_vals)

        size_elem = elem.find("./geometry/box/size")
        if size_elem is not None and size_elem.text:
            size_vals = size_elem.text.split()
            if len(size_vals) == 3:
                size_vals[2] = f"{float(size_vals[2]) * scale:.6f}".rstrip("0").rstrip(".")
                size_elem.text = " ".join(size_vals)

    tmp = tempfile.NamedTemporaryFile(prefix="huro_stairs_", suffix=".sdf", delete=False)
    tmp_path = tmp.name
    tmp.close()
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    return tmp_path


def _launch_setup(context, *args, **kwargs):
    step_height = float(LaunchConfiguration("step_height").perform(context))

    world_name = "empty_world"
    model_name = "go2"
    joint_order = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    huro_share = get_package_share_directory("huro")
    huro_prefix = get_package_prefix("huro")
    huro_lib = os.path.join(huro_prefix, "lib", "huro")
    ws_root = os.path.dirname(os.path.dirname(huro_prefix))

    def _resolve_script(script_name: str) -> str:
        candidates = [
            os.path.join(huro_lib, script_name),
            os.path.join(ws_root, "src", "huro", "apps", script_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Could not locate {script_name}. Tried: {candidates}"
        )

    gz_base_tf_script = _resolve_script("gz_base_tf_publisher.py")
    gz_state_adapter_script = _resolve_script("gz_go2_state_adapter.py")
    urdf_path = os.path.join(
        huro_share,
        "resources",
        "description_files",
        "urdf",
        "go2",
        "go2_gz.urdf",
    )
    rviz_config = os.path.join(huro_share, "resources", "rviz", "go2.rviz")
    world_template_path = os.path.join(huro_share, "resources", "worlds", "stairs.sdf")
    world_path = _scale_stairs_world(world_template_path, step_height)
    bridge_cfg_path = os.path.join(
        huro_share, "resources", "bridge", "go2_sim_gz_bridge.yaml"
    )
    ros_gz_sim_launch = (
        get_package_share_directory("ros_gz_sim") + "/launch/gz_sim.launch.py"
    )

    with open(bridge_cfg_path, "r", encoding="utf-8") as f:
        bridge_cfg = yaml.safe_load(f) or {}

    fmt_values = {"world_name": world_name, "model_name": model_name}

    def _fmt(text: str) -> str:
        return str(text).format(**fmt_values)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        arguments=[urdf_path],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ros_gz_sim_launch),
        launch_arguments={"gz_args": f"{world_path} -r"}.items(),
    )

    create_from_robot_description = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-topic",
                    "robot_description",
                    "-name",
                    model_name,
                    "-z",
                    "0.40",
                ],
                output="screen",
            )
        ],
    )

    gz_bridge_args = [_fmt(arg) for arg in bridge_cfg.get("arguments", [])]

    for jn in joint_order:
        gz_bridge_args.append(
            f"/model/{model_name}/joint/{jn}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double"
        )

    gz_bridge_remappings = [
        (_fmt(item["from"]), _fmt(item["to"]))
        for item in bridge_cfg.get("remappings", [])
    ]

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="go2_gz_bridge",
        arguments=gz_bridge_args,
        remappings=gz_bridge_remappings,
        output="screen",
    )

    gz_base_tf_publisher = ExecuteProcess(
        cmd=[
            "python3",
            gz_base_tf_script,
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "input_topic:=/gz_pose_tf",
            "-p",
            "world_frame:=world",
            "-p",
            "base_frame:=base",
            "-p",
            f"model_name:={model_name}",
        ],
        output="screen",
    )

    gz_go2_state_adapter = ExecuteProcess(
        cmd=[
            "python3",
            gz_state_adapter_script,
            "--ros-args",
            "-p",
            "use_sim_time:=true",
        ],
        output="screen",
    )

    # Fallback TF for Gazebo lidar frame naming used by some bridges.
    lidar_frame_fallback_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_frame_fallback_tf",
        arguments=[
            "0.28945",
            "0",
            "-0.046825",
            "0",
            "2.8782",
            "0",
            "base",
            "go2/base/go2_lidar",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return [
        robot_state_publisher,
        gz_sim,
        create_from_robot_description,
        gz_bridge,
        gz_base_tf_publisher,
        gz_go2_state_adapter,
        lidar_frame_fallback_tf,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "step_height",
                default_value=str(DEFAULT_STEP_HEIGHT),
                description="Single step height in meters (original world uses 0.06).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

