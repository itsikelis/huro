import os
import launch
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory("huro")
    rviz_launch = IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            share_dir + "/launch/g1_rviz.launch.py"
        )
    )

    ## HURo Sim Node ##
    config = os.path.join(share_dir, "config", "g1_sim_params.yaml")
    sim_node = Node(
        package="huro",
        executable="sim_g1",
        name="sim_g1",
        parameters=[config],
    )

    return launch.LaunchDescription(
        [
            rviz_launch,
            sim_node,
        ]
    )
