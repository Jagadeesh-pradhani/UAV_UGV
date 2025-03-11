from launch import LaunchDescription
import launch_ros.actions
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    slam_examples_dir = FindPackageShare(package="sjtu_drone_bringup").find("sjtu_drone_bringup")

    declare_octomap_res = DeclareLaunchArgument(
        name="octomap_res",
        default_value="0.05",
        description="Resolution of octomap to load.",
    )

    return LaunchDescription(
        [
            declare_octomap_res,
            launch_ros.actions.Node(
                package="octomap_server",
                executable="octomap_server_node",
                name="octomap_server",
                output="screen",
                parameters=[
                    slam_examples_dir + "/config/online_params.yaml",
                    {
                        "frame_id": "map",
                        "use_height_map": True,
                        "resolution": LaunchConfiguration("octomap_res"),
                    },
                ],
                remappings=[("/cloud_in", "/simple_drone/front/points")],
            ),
        ]
    )
