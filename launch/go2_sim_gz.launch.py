import os
import yaml

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory


import xml.etree.ElementTree as ET
DEFAULT_SPAWN_X = 0.0
DEFAULT_SPAWN_Y = 0.0
DEFAULT_SPAWN_Z = 0.40


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
    world_path = os.path.join(huro_share, "resources", "worlds", "factory.sdf")
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

    create_from_robot_description = TimerAction(
        period=1.0,
        actions=[
            Node(
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
            'Grid/RayTracing': 'true',
            'Grid/3D': 'false',
            # Ne pas marquer escaliers / terrain comme obstacles
            'Grid/MaxObstacleHeight': '0.5',
            'Grid/NormalsSegmentation': 'false',
            'Grid/MaxGroundAngle': '45',       # tolère les pentes
        }],
        remappings=[
            ('scan_cloud', '/utlidar/cloud'),
            ('odom', '/odom'),
            # ('grid_prob_map', '/map'),
            # ('/grid_prob_map', '/map'),
        ]
    )
    
    return [
        robot_state_publisher,
        gz_sim,
        create_from_robot_description,
        gz_bridge,
        gz_go2_state_adapter,
        lidar_frame_fallback_tf,
        imu_frame_fallback_tf,
        rviz_node,
        TimerAction(period=2.0, actions=[odom_node]),
        TimerAction(period=4.0, actions=[slam_node]),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_launch_setup)])

