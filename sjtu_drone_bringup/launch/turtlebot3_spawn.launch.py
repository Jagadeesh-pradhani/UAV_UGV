import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, LogInfo, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetParameter, SetRemap, PushRosNamespace

import xml.etree.ElementTree as ET
import yaml

package_name = 'sjtu_drone_bringup'
launch_file_dir = PathJoinSubstitution([FindPackageShare(package_name), 'launch'])
pkg_gazebo_ros = FindPackageShare('gazebo_ros')


def group_action(context : LaunchContext):

    num_robots = int(context.launch_configurations['num_robots'])
    visualise_sensors = context.launch_configurations['visualise_sensors'].lower()
    odometry_source = context.launch_configurations['odometry_source']
    sensor_noise = eval(context.launch_configurations['sensor_noise'].lower().capitalize())
    initial_pose_package = context.launch_configurations['initial_pose_package']
    initial_pose_file = context.launch_configurations['initial_pose_file']

    sdf_path = os.path.join(get_package_share_directory(package_name), 'models', 'waffle_pi', 'model.sdf')
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    for node in root.iter("visualize"):
        for element in node.iter():
            element.text = visualise_sensors

    for node in root.iter("odometry_source"):
        for element in node.iter():
            if odometry_source == "ENCODER":
                element.text = "0"
            elif odometry_source == "WORLD":
                element.text = "1"

    if sensor_noise == False:

        for sensor in root.findall('.//sensor'):
            for imu in sensor.findall('.//imu'):
                sensor.remove(imu)

        for ray in root.findall('.//ray'):
            for noise in ray.findall('.//noise'):
                ray.remove(noise)

        for camera in root.findall('.//camera'):
            for noise in camera.findall('.//noise'):
                camera.remove(noise)

    robot_sdf = os.path.join(get_package_share_directory(package_name), 'models', 'waffle_pi', 'model.sdf')

    with open(robot_sdf, 'w') as f:
        tree.write(f, encoding='unicode')

    yaml_path = os.path.join(get_package_share_directory(initial_pose_package), initial_pose_file)

    print('pose: ' + yaml_path)

    with open(yaml_path, 'r') as f:
        configuration = yaml.safe_load(f)

    initial_poses = configuration[num_robots]

    with open(context.launch_configurations['rviz_windows'], 'r') as f:
        configuration = yaml.safe_load(f)

    rviz_windows = configuration[num_robots]

    bringup_cmd_group = []

    for robot_name, init_pose in initial_poses.items():
        group = GroupAction([

            PushRosNamespace(robot_name),
            SetRemap('/tf', 'tf'),
            SetRemap('/tf_static', 'tf_static'),

            LogInfo(msg=['Launching namespace=', robot_name, ' init_pose=', str(init_pose)]),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([launch_file_dir, 'rviz_launch.py'])),
                condition=IfCondition(context.launch_configurations['use_rviz']),
                launch_arguments={'rviz_config': context.launch_configurations['rviz_config'],
                                  'window_x': str(rviz_windows[robot_name]['window_x']),
                                  'window_y': str(rviz_windows[robot_name]['window_y']),
                                  'window_width': str(rviz_windows[robot_name]['window_width']),
                                  'window_height': str(rviz_windows[robot_name]['window_height'])}.items()),

            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([
                    launch_file_dir,
                    'spawn_robot_launch.py'])),
                launch_arguments={'x_pose': TextSubstitution(text=str(init_pose['x'])),
                                  'y_pose': TextSubstitution(text=str(init_pose['y'])),
                                  'yaw': TextSubstitution(text=str(init_pose['yaw'])),
                                  'robot_sdf': robot_sdf}.items())
        ])
    
        bringup_cmd_group.append(group)

    return bringup_cmd_group

def generate_launch_description():

    num_robots = LaunchConfiguration('num_robots')

    declare_initial_pose_file = DeclareLaunchArgument(
        'initial_pose_file',
        default_value='config/initial_poses.yaml',
        description="Location of initial pose yaml file relative to the package in 'initial_pose_package'"
    )

    declare_initial_pose_package = DeclareLaunchArgument(
        'initial_pose_package',
        default_value='sjtu_drone_bringup',
        description="Package name for finding the file 'config/initial_poses.yaml'"
        )

    declare_num_robots_cmd = DeclareLaunchArgument(
        'num_robots',
        default_value='1',
        description='Number of robots to spawn')
    
    declare_visualise_sensors_cmd = DeclareLaunchArgument(
        'visualise_sensors',
        default_value='false',
        description='Whether to visualise sensors in Gazebo')
    
    declare_odometry_source_cmd = DeclareLaunchArgument(
        'odometry_source',
        default_value='ENCODER',
        description='Odometry source - ENCODER or WORLD')
    
    declare_sensor_noise_cmd = DeclareLaunchArgument(
        'sensor_noise',
        default_value='false',
        description='Whether to enable sensor noise (applies to camera, LiDAR, and IMU)')
    
    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Whether to start RViz')
    
    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([FindPackageShare(package_name), 'rviz', 'namespaced.rviz']),
        description='Full path to the RViz config file to use')
    
    declare_rviz_windows_cmd = DeclareLaunchArgument(
        'rviz_windows',
        default_value=PathJoinSubstitution([FindPackageShare(package_name), 'config', 'rviz_windows.yaml']),
        description='Full path to the RViz windows YAML file to use')        
    
    
    


    start_tf_relay_cmd = Node(
        package='tf_relay',
        executable='relay',
        output='screen',
        arguments=['robot', num_robots])



    bringup_cmd_group = OpaqueFunction(function=group_action)
        
    ld = LaunchDescription()

    ld.add_action(SetParameter(name='use_sim_time', value=True))

    # Declare the launch options
    ld.add_action(declare_num_robots_cmd)
    ld.add_action(declare_visualise_sensors_cmd)
    ld.add_action(declare_odometry_source_cmd)
    ld.add_action(declare_sensor_noise_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_rviz_windows_cmd)

 


    # Add launch initial pose option
    ld.add_action(declare_initial_pose_package)
    ld.add_action(declare_initial_pose_file)


    ld.add_action(start_tf_relay_cmd)

    ld.add_action(bringup_cmd_group)

    return ld
