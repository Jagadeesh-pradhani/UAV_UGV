# UAV_UGV

```bash
ros2 topic pub /drone_goal geometry_msgs/msg/Point "{x: 14.0, y: 0.0, z: 5.0}" -1

-1.0 4.0 3.0 
-4.0 4.0 3.0
-1.0 0.0 3.0 
-4.0 0.0 3.0 
-4.0 -4.0 3.0
1.0 0.0 3.0


ros2 topic pub /drone_goal geometry_msgs/msg/Point "{x: -1.0, y: 4.0, z: 3.0}" -1
ros2 topic pub /drone_goal geometry_msgs/msg/Point "{x: -4.0, y: 4.0, z: 3.0}" -1

```

## Steps

```
ros2 launch sjtu_drone_bringup sjtu_drone_bringup_slam.launch.py
ros2 launch sjtu_drone_bringup octomap_online.launch.py
python3 astar_3d.py
ros2 run sjtu_drone_control fellow_path_controller
```