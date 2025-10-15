import os
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

urdf = "go2/go2.urdf"
rviz_config = "go2.rviz"


def generate_launch_description():

    ## Robot State Publisher ##
    # Find and load robot description
    urdf_path = os.path.join(
        get_package_share_directory("huro") + "/resources/description_files/urdf/",
        urdf,
    )
    with open(urdf_path, "r") as infp:
        robot_desc = infp.read()

    # Create robot state publisher node
    state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc}],
        arguments=[urdf_path],
    )

    ## RViz ##
    # Find rviz path
    rviz_file_path = os.path.join(
        get_package_share_directory("huro") + "/resources/rviz/",
        rviz_config,
    )
    # Create rviz node
    rviz_node = Node(
        package="rviz2",
        namespace="",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d" + rviz_file_path,
        ],
    )

    ## HURo Node ##
    core_node = Node(package="huro", executable="root_go2", name="root_go2")

    # # Livox Lidar Launch File
    # lidar_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         PathJoinSubstitution(
    #             [
    #                 FindPackageShare("livox_ros_driver2"),
    #                 "launch_ROS2",
    #                 "rviz_MID360_launch.py",
    #             ]
    #         )
    #     ),
    # )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),
            core_node,
            state_pub_node,
            rviz_node,
        ]
    )
