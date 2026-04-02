import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    huro_share = get_package_share_directory("huro")
    urdf_path = os.path.join(
        huro_share,
        "resources",
        "description_files",
        "urdf",
        "go2",
        "go2_gz.urdf",
    )
    rviz_config = os.path.join(huro_share, "resources", "rviz", "go2.rviz")
    ros_gz_sim_launch = (
        get_package_share_directory("ros_gz_sim") + "/launch/gz_sim.launch.py"
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        arguments=[urdf_path],
        output="screen",
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        arguments=[urdf_path],
        output="screen",
    )

    static_world_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_world_to_base",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base"],
        output="screen",
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ros_gz_sim_launch),
        launch_arguments={"gz_args": "empty.sdf -r"}.items(),
    )

    create_from_robot_description = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=["-topic", "robot_description", "-z", "0.40"],
                output="screen",
            )
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription(
        [
            robot_state_publisher,
            joint_state_publisher,
            static_world_to_base,
            gz_sim,
            create_from_robot_description,
            rviz_node,
        ]
    )

