import os
import tempfile
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory


import xml.etree.ElementTree as ET
DEFAULT_SPAWN_X = -0.0
DEFAULT_SPAWN_Y = -0.0
DEFAULT_SPAWN_Z = 0.40
DEFAULT_SPAWN_YAW = 0.0
DEFAULT_STAIRS_STEP_HEIGHT = 0.06

_STAIRS_HEIGHT_MULTIPLIERS = [
    0.5,
    1.5,
    2.5,
    3.5,
    4.5,
    5.5,
    5.5,
    5.5,
    4.5,
    3.5,
    2.5,
    1.5,
    0.5,
]


def _resolve_stairs_step_height(context) -> float:
    raw_value = LaunchConfiguration("stairs_step_height").perform(context)
    try:
        value = float(raw_value)
        if value <= 0.0:
            raise ValueError
        return value
    except Exception:
        print(
            f"[huro] Invalid stairs_step_height='{raw_value}'. "
            f"Falling back to {DEFAULT_STAIRS_STEP_HEIGHT}."
        )
        return DEFAULT_STAIRS_STEP_HEIGHT


def _generate_world_with_custom_stairs_height(template_world_path: str, step_height: float) -> str:
    tree = ET.parse(template_world_path)
    root = tree.getroot()

    stairs_link = root.find(".//model[@name='stairs_up_down']/link[@name='stairs_link']")
    if stairs_link is None:
        return template_world_path

    for step_idx, z_mult in enumerate(_STAIRS_HEIGHT_MULTIPLIERS, start=1):
        x_pos = 0.84 + (step_idx - 1) * 0.28
        z_pos = z_mult * step_height
        pose_text = f"{x_pos:.2f} 0 {z_pos:.6f} 0 0 0"
        size_text = f"0.30 4.80 {step_height:.6f}"

        for tag_name, suffix in (("collision", "col"), ("visual", "vis")):
            step_elem = stairs_link.find(
                f"{tag_name}[@name='step_{step_idx:02d}_{suffix}']"
            )
            if step_elem is None:
                continue

            pose_elem = step_elem.find("pose")
            if pose_elem is not None:
                pose_elem.text = pose_text

            size_elem = step_elem.find("geometry/box/size")
            if size_elem is not None:
                size_elem.text = size_text

    temp_world_path = os.path.join(
        tempfile.gettempdir(), f"huro_factory_step_{step_height:.4f}.sdf"
    )
    tree.write(temp_world_path, encoding="utf-8", xml_declaration=True)
    return temp_world_path


def _get_world_name(world_path: str) -> str:
    try:
        root = ET.parse(world_path).getroot()
        world_elem = root.find("world")
        if world_elem is not None and world_elem.get("name"):
            return world_elem.get("name")
    except Exception:
        pass
    return "world"

def _launch_setup(context, *args, **kwargs):
    spawn_x = DEFAULT_SPAWN_X
    spawn_y = DEFAULT_SPAWN_Y
    spawn_z = DEFAULT_SPAWN_Z
    spawn_yaw = DEFAULT_SPAWN_YAW
    stairs_step_height = _resolve_stairs_step_height(context)

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

    gz_state_adapter_script = os.path.join(huro_lib, "gz_go2_state_adapter.py")
    urdf_path = os.path.join(huro_share,"resources","description_files","urdf","go2","go2_gz.urdf",)
    rviz_config = os.path.join(huro_share, "resources", "rviz", "go2.rviz")
    nav2_config = os.path.join(huro_share, 'resources', 'nav2', 'nav2_params.yaml')

    world_template_path = os.path.join(huro_share, "resources", "worlds", "factory.sdf")
    world_path = _generate_world_with_custom_stairs_height(
        world_template_path, stairs_step_height
    )
    world_name = _get_world_name(world_path)
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

    create_from_robot_description = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            model_name,
            "-x",
            str(spawn_x),
            "-y",
            str(spawn_y),
            "-z",
            str(spawn_z),
            "-Y",
            str(spawn_yaw),
        ],
        output="screen",
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

    # Fallback TF for Gazebo IMU frame naming used by some bridges/plugins.
    imu_frame_fallback_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="imu_frame_fallback_tf",
        arguments=[
            "-0.02557",
            "0",
            "0.04232",
            "0",
            "0",
            "0",
            "base",
            "go2/base/go2_imu_sensor",
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
    
    odom_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        name='icp_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base',
            'odom_frame_id': 'odom',
            # Use odom as the single TF source for odom->base.
            'publish_tf': True,
            'wait_imu_to_init': True,
            # Allow a small TF wait to absorb sim timestamp jitter.
            'wait_for_transform': 0.2,
            # Avoid invalid (non-normalized) null quaternions on tracking loss.
            'publish_null_when_lost': False,
            # Keep lidar odometry independent from IMU frame naming issues.
            'subscribe_imu': True,
            'use_sim_time': True,
            'Icp/PointToPlane': 'true',
            'Icp/Iterations': '30',
            'Icp/VoxelSize': '0.15',
            'Icp/CorrespondenceRatio': '0.15',
            'Icp/MaxTranslation': '1.5',
            'Icp/MaxRotation': '1.57',
            'Reg/Force3DoF': 'true',
            'Odom/ResetCountdown': '1',
        }],
        remappings=[
            ('/scan_cloud', '/utlidar/cloud'),
            ('scan_cloud', '/utlidar/cloud'),
            ('/odom', '/odom'),
            ('odom', '/odom'),
            ('/odom_info', '/odom_info'),
            ('odom_info', '/odom_info'),
        ]
    )
    slam_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[{
            'frame_id': 'base',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            'subscribe_rgb': False,
            'subscribe_depth': False,
            'subscribe_scan': False,
            'subscribe_scan_cloud': True,
            'subscribe_lidar': True,
            'subscribe_imu': True,
            'use_sim_time': True,
            # Avoid odom->base future extrapolation on slight TF latency.
            'wait_for_transform': 0.2,
            'topic_queue_size': 50,
            'sync_queue_size': 50,
            # Keep SLAM map strictly planar (no roll/pitch/z drift in map frame).
            # 'Reg/Force3DoF': 'true',
            # 'Optimizer/Slam2D': 'true',
            # 'Grid/RayTracing': 'true',
            # 'Grid/3D': 'true',
            # Ne pas marquer escaliers / terrain comme obstacles
            # 'Grid/MaxObstacleHeight': '1.0',
            # 'Grid/NormalsSegmentation': 'false',
            # 'Grid/MaxGroundAngle': '45',       # tolère les pentes
        }],
        remappings=[
            ('scan_cloud', '/utlidar/cloud'),
            ('odom', '/odom'),
        ]
    )
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            get_package_share_directory('nav2_bringup') + '/launch/navigation_launch.py'
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'params_file': nav2_config,
            'map_subscribe_transient_local': 'True',
            'autostart': 'True',
        }.items(),
        
    )
   
    return [
        robot_state_publisher,
        gz_sim,
        create_from_robot_description,
        gz_bridge,
        gz_go2_state_adapter,
        lidar_frame_fallback_tf,
        imu_frame_fallback_tf,
        TimerAction(period=2.0, actions=[odom_node]),
        TimerAction(period=4.0, actions=[slam_node]),
        TimerAction(period=6.0, actions=[nav2_launch]),
        TimerAction(period=8.0, actions=[rviz_node]),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "stairs_step_height",
                default_value=str(DEFAULT_STAIRS_STEP_HEIGHT),
                description=(
                    "Height of each step in meters for the stairs_up_down model "
                    "in factory.sdf."
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

