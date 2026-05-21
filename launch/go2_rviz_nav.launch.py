    
import os
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

urdf = "go2/go2.urdf"
rviz_config = "go2.rviz"
nav2 = 'nav2_params.yaml'


def generate_launch_description():

    ## Robot State Publisher ##
    # Find and load robot description
    urdf_path = os.path.join(
        get_package_share_directory("huro") + "/resources/description_files/urdf/",
        urdf,
    )
    nav2_config = os.path.join(get_package_share_directory("huro"), 'resources', 'nav2', nav2)

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
            'wait_for_transform': 0.2,
            'topic_queue_size': 50,
            'sync_queue_size': 50,
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

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),

            core_node,
            state_pub_node,
            TimerAction(period=2.0, actions=[odom_node]),
            TimerAction(period=4.0, actions=[slam_node]),
            TimerAction(period=6.0, actions=[nav2_launch]),
            TimerAction(period=8.0, actions=[rviz_node]),
        ]
    )

