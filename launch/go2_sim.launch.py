import launch
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    foo_dir = get_package_share_directory("huro")
    rviz_launch = IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            foo_dir + "/launch/go2_rviz.launch.py"
        )
    )

    ## HURo Sim Node ##
    sim_node = Node(package="huro", executable="sim_go2", name="sim_go2")

    return launch.LaunchDescription(
        [
            rviz_launch,
            sim_node,
        ]
    )
