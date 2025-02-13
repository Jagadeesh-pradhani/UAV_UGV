#!/usr/bin/env python3
# Copyright 2023 Georg Novotny
#
# Licensed under the GNU GENERAL PUBLIC LICENSE, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, GroupAction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import PushRosNamespace, SetRemap, Node


def get_teleop_controller(context, *_, **kwargs) -> Node:
    controller = context.launch_configurations["controller"]
    namespace = kwargs["model_ns"]
    
    if controller == "joystick":
        node = Node(
            package="sjtu_drone_control",
            executable="teleop_joystick",
            namespace=namespace,
            output="screen",
        )

    else:
        node = Node(
            package="sjtu_drone_control",
            executable="teleop",
            namespace=namespace,
            output="screen",
            prefix="xterm -e",
        )

    return [node]

def generate_launch_description():
    sjtu_drone_bringup_path = get_package_share_directory('sjtu_drone_bringup')
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')

    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file')
    map_dir = LaunchConfiguration('map')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    declare_map_dir_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(
            sjtu_drone_bringup_path,
            'map',
            'map_2.yaml'),
        description='Full path to map file to load'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            sjtu_drone_bringup_path,
            'params',
            'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    x_pose = DeclareLaunchArgument(
        "x_pose",
        default_value="1.0",
        description="Initial x position for the turtlebot in Gazebo"
    )
    y_pose = DeclareLaunchArgument(
        "y_pose",
        default_value="1.0",
        description="Initial y position for the turtlebot in Gazebo"
    )
    
    rviz_path = os.path.join(
        sjtu_drone_bringup_path, "rviz", "mix3.rviz"
    )
    
    yaml_file_path = os.path.join(
        get_package_share_directory('sjtu_drone_bringup'),
        'config', 'drone.yaml'
    )

    # Load namespace from YAML file
    model_ns = "drone"
    with open(yaml_file_path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        model_ns = yaml_dict["namespace"]
    
    # Robot controllers

    # Return the LaunchDescription with valid actions
    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_map_dir_cmd,
        declare_params_file_cmd,
        x_pose,
        y_pose,
        DeclareLaunchArgument(
            "controller",
            default_value="keyboard",
            description="Type of controller: keyboard (default) or joystick",
        ),
        # Node(
        #     package="rviz2",
        #     executable="rviz2",
        #     name="rviz2",
        #     arguments=["-d", rviz_path],
        #     output="screen",
        # ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sjtu_drone_bringup_path, 'launch', 'sjtu_drone_gazebo.launch.py')
            )
        ),
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         os.path.join(sjtu_drone_bringup_path, 'launch', 'turtlebot3_bringup.launch.py')
        #     )
        # ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
            ),
            launch_arguments={
                'x_pose': LaunchConfiguration('x_pose'),
                'y_pose': LaunchConfiguration('y_pose')
            }.items()
        ),
        

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sjtu_drone_bringup_path, 'launch', 'navigation2.launch.py')
            ),
            launch_arguments={
                'map': map_dir,
                'use_sim_time': use_sim_time,
                'params_file': params_file}.items(),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0", "world", "map"],
            output="screen"
        ),
        
        Node(
            package='sjtu_drone_control',
            executable='fellow_path_controller',
            name='fellow_path_controller',
            output='screen',
        ),
        Node(
            package='prm',
            executable='map_publisher',
            name='map_publisher',
            output='screen',
        )




        # Node(
        #     package='joy',
        #     executable='joy_node',
        #     name='joy',
        #     namespace=model_ns,
        #     output='screen',
        # ),
        # OpaqueFunction(
        #     function=get_teleop_controller,
        #     kwargs={'model_ns': model_ns},
        # ),
    ])
