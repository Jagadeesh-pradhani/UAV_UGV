import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix
import yaml

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, Shutdown, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter, SetRemap, PushRosNamespace, RosTimer
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.event_handlers import OnProcessStart


def generate_launch_description():

    package_name = 'solution'

    num_robots = LaunchConfiguration('num_robots')
    random_seed = LaunchConfiguration('random_seed')

    declare_num_robots_cmd = DeclareLaunchArgument(
        'num_robots',
        default_value='1',
        description='Number of robots to spawn')
    
    declare_random_seed_cmd = DeclareLaunchArgument(
        'random_seed',
        default_value='0',
        description='Random number seed for item manager')
    

    

    


    rviz_config = PathJoinSubstitution([FindPackageShare('assessment'), 'rviz', 'namespaced.rviz'])
    rviz_windows = PathJoinSubstitution([FindPackageShare('assessment'), 'config', 'rviz_windows.yaml'])
    # rviz_windows = PathJoinSubstitution([FindPackageShare(package_name), 'config', 'custom_rviz_windows.yaml'])

    assessment_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('sjtu_drone_bringup'),
                'launch',
                'turtlebot3_custom.launch.py'
                ])
        ),
        launch_arguments={'num_robots': num_robots,
                          'visualise_sensors': 'false',
                          'odometry_source': 'ENCODER',
                          'sensor_noise': 'false',
                          'use_rviz': 'true',
                          'rviz_config': rviz_config,
                          'rviz_windows': rviz_windows,
                          'limit_real_time_factor': 'true',
                          'wait_for_items': 'false',
                          # 'extra_gazebo_args': '--verbose',
                          }.items()
    )



    ld = LaunchDescription()

    ld.add_action(SetParameter(name='use_sim_time', value=True))

    ld.add_action(declare_num_robots_cmd)
    ld.add_action(declare_random_seed_cmd)

    ld.add_action(assessment_cmd)


    return ld
