import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory


def generate_launch_description():
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
    world_path = os.path.join(huro_share, "resources", "worlds", "empty.sdf")
    ros_gz_sim_launch = (
        get_package_share_directory("ros_gz_sim") + "/launch/gz_sim.launch.py"
    )

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

    gz_bridge_args = [
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        f"/world/{world_name}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        f"/world/{world_name}/model/{model_name}/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        f"/world/default/model/{model_name}/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        f"/model/{model_name}/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        f"/model/{model_name}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        f"/world/{world_name}/model/{model_name}/link/imu/sensor/go2_imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        f"/world/default/model/{model_name}/link/imu/sensor/go2_imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        f"/model/{model_name}/link/imu/sensor/go2_imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        "/livox/lidar@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        "/lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        "/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        f"/world/{world_name}/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        f"/world/default/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        f"/world/{world_name}/model/{model_name}/link/base/sensor/go2_lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        f"/world/default/model/{model_name}/link/base/sensor/go2_lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        f"/world/{world_name}/model/{model_name}/link/Head_lower/sensor/go2_lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        f"/world/default/model/{model_name}/link/Head_lower/sensor/go2_lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        f"/world/{world_name}/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        f"/world/default/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        f"/world/{world_name}/model/{model_name}/link/base/sensor/go2_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        f"/world/default/model/{model_name}/link/base/sensor/go2_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        f"/world/{world_name}/model/{model_name}/link/Head_lower/sensor/go2_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
        f"/world/default/model/{model_name}/link/Head_lower/sensor/go2_lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
    ]

    for jn in joint_order:
        gz_bridge_args.append(
            f"/model/{model_name}/joint/{jn}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double"
        )

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="go2_gz_bridge",
        arguments=gz_bridge_args,
        remappings=[
            (f"/world/{world_name}/model/{model_name}/joint_state", "/joint_states_gz"),
            (f"/world/default/model/{model_name}/joint_state", "/joint_states_gz"),
            (f"/model/{model_name}/joint_state", "/joint_states_gz"),
            ("/imu", "/imu"),
            (f"/model/{model_name}/imu", "/imu"),
            (
                f"/world/{world_name}/model/{model_name}/link/imu/sensor/go2_imu_sensor/imu",
                "/imu",
            ),
            (
                f"/world/default/model/{model_name}/link/imu/sensor/go2_imu_sensor/imu",
                "/imu",
            ),
            (f"/model/{model_name}/link/imu/sensor/go2_imu_sensor/imu", "/imu"),
            ("/lidar", "/livox/scan"),
            ("/lidar/points", "/livox/lidar"),
            (
                f"/world/{world_name}/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/points",
                "/livox/lidar",
            ),
            (
                f"/world/default/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/points",
                "/livox/lidar",
            ),
            (
                f"/world/{world_name}/model/{model_name}/link/base/sensor/go2_lidar/points",
                "/livox/lidar",
            ),
            (
                f"/world/default/model/{model_name}/link/base/sensor/go2_lidar/points",
                "/livox/lidar",
            ),
            (
                f"/world/{world_name}/model/{model_name}/link/Head_lower/sensor/go2_lidar/points",
                "/livox/lidar",
            ),
            (
                f"/world/default/model/{model_name}/link/Head_lower/sensor/go2_lidar/points",
                "/livox/lidar",
            ),
            (
                f"/world/{world_name}/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/scan",
                "/livox/scan",
            ),
            (
                f"/world/default/model/{model_name}/link/utlidar_lidar/sensor/go2_lidar/scan",
                "/livox/scan",
            ),
            (
                f"/world/{world_name}/model/{model_name}/link/base/sensor/go2_lidar/scan",
                "/livox/scan",
            ),
            (
                f"/world/default/model/{model_name}/link/base/sensor/go2_lidar/scan",
                "/livox/scan",
            ),
            (
                f"/world/{world_name}/model/{model_name}/link/Head_lower/sensor/go2_lidar/scan",
                "/livox/scan",
            ),
            (
                f"/world/default/model/{model_name}/link/Head_lower/sensor/go2_lidar/scan",
                "/livox/scan",
            ),
            (f"/world/{world_name}/dynamic_pose/info", "/gz_pose_tf"),
        ],
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
            "-p",
            "joint_state_topic:=/joint_states_gz",
            "-p",
            "joint_state_out_topic:=/joint_states",
            "-p",
            "lowcmd_topic:=/lowcmd",
            "-p",
            "tf_topic:=/tf",
            "-p",
            "imu_topic:=/imu",
            "-p",
            "lowstate_topic:=/lowstate",
            "-p",
            "sportmode_topic:=/sportmodestate",
            "-p",
            "world_frame:=world",
            "-p",
            "base_frame:=base",
            "-p",
            f"model_name:={model_name}",
            "-p",
            "lowstate_quat_wxyz:=true",
        ],
        output="screen",
    )

    # Fallback TF for Gazebo lidar frame naming used by some bridges.
    lidar_frame_fallback_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_frame_fallback_tf",
        arguments=["0", "0", "0", "0", "0", "0", "base", "go2/base/go2_lidar"],
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

    return LaunchDescription(
        [
            robot_state_publisher,
            gz_sim,
            create_from_robot_description,
            gz_bridge,
            gz_base_tf_publisher,
            gz_go2_state_adapter,
            lidar_frame_fallback_tf,
            rviz_node,
        ]
    )

