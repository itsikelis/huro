import os
import launch
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory("huro")
    rviz_launch = IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            share_dir + "/launch/go2_rviz.launch.py"
        )
    )
    with open(urdf_path, "r") as infp:
        robot_desc = infp.read()


    ## HURo Node ##
    core_node = Node(package="huro", executable="root_go2", name="root_go2")

    ## Joy Node ##
    joy_node = Node(package="joy", executable="joy_node", name="joy_node")

    ## HURo Sim Node ##
    config = os.path.join(share_dir, "config", "go2_sim_params.yaml")
    sim_node = Node(
        package="huro",
        executable="sim_go2",
        name="sim_go2",
        parameters=[config],
    )

    # Create robot state publisher node
    state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc}],
        arguments=[urdf_path],
    )

    state_pub_commands_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        name='robot_state_publisher_controls',
        parameters=[{
            'robot_description': robot_desc,
            'frame_prefix': "commands/",
        }],
        remappings=[
            ('robot_description', '/robot_description_commands'),  # Remap the topic
            ('joint_states', '/joint_commands'),
        ],
    )


    ## RViz ##
    # Find rviz path
    rviz_file_path = os.path.join(
        get_package_share_directory("huro") + "/resources/rviz/",
        "go2_sim.rviz",
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

    return launch.LaunchDescription(
        [
            core_node,
            sim_node,
            state_pub_node,
            state_pub_commands_node,
            rviz_node,
            joy_node,
        ]
    )
